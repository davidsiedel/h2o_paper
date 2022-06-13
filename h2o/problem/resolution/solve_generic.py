import numpy as np

from h2o.problem.problem import Problem, clean_res_dir
from h2o.problem.material import Material
from h2o.h2o import *

from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sys
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=2,suppress=True, threshold=sys.maxsize, formatter=None)

from h2o.problem.output import create_output_txt


def solve_newton(problem: Problem, material: Material, solver_type: SolverType):
    clean_res_dir(problem.res_folder_path)
    problem.create_output(problem.res_folder_path)
    output_file_path = os.path.join(problem.res_folder_path, "output.txt")
    create_output_txt(output_file_path, problem, material)
    _dx: int = problem.field.field_dimension
    _fk: int = problem.finite_element.face_basis_k.dimension
    _cl: int = problem.finite_element.cell_basis_l.dimension
    external_forces_coefficient: float = 1.0
    # ---SET SYSTEM SIZE
    _constrained_system_size, _system_size = problem.get_total_system_size()
    faces_unknown_vector: ndarray = np.zeros((_constrained_system_size), dtype=real)
    faces_unknown_vector_previous_step: ndarray = np.zeros((_constrained_system_size), dtype=real)
    computation_time_start = time.time()
    with open(output_file_path, "a") as outfile:
        outfile.write(
            "----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** COMPUTATION\n")
        outfile.write("+ SYSTEM SIZE : {}\n".format(_system_size))
        print("+ SYSTEM SIZE : {}".format(_system_size))
        outfile.write("+ RESOLUTION_ALGORITHM : {}".format(solver_type.name))
        outfile.write("\n")
        print("+ RESOLUTION_ALGORITHM : {}".format(solver_type.name))
        qp_count: int = 0
        qp_final: int = 0
        # --- CHECK QUADRATURE POINTS NUMBER CONSISTENCY
        for element_index, element in enumerate(problem.elements):
            qp_count += len(element.quad_p_indices)
            if element_index == len(problem.elements) - 1:
                qp_final = element.quad_p_indices[-1]
        if qp_count != problem.mesh.number_of_cell_quadrature_points_in_mesh:
            raise ArithmeticError("the number of quadrature points is not right : qp in mesh {} | qp_count {}".format(problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_count))
        if qp_final != problem.mesh.number_of_cell_quadrature_points_in_mesh - 1:
            raise ArithmeticError("the number of quadrature points is not right : qp in mesh {} | qp_final {}".format(problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_final))
        print("+ NUMBER OF INTEGRATION POINTS : {}".format(problem.mesh.number_of_cell_quadrature_points_in_mesh))
        outfile.write("+ NUMBER OF INTEGRATION POINTS IN MESH : {}".format(problem.mesh.number_of_cell_quadrature_points_in_mesh))
        outfile.write("\n")
        # --- TIME STEP INIT
        time_step_index: int = 0
        stab_param_global: float = material.stabilization_parameter
        stab_param_local: float = material.stabilization_parameter
        time_step_temp: float = problem.time_steps[0]
        local_external_forces_coefficient = 1.
        while time_step_index < len(problem.time_steps):
            step_time_start = time.time()
            time_step: float = problem.time_steps[time_step_index]
            material.set_temperature()
            # mgis_bv.setExternalStateVariable(material.mat_data.s0, "Temperature", material.temperature)
            # mgis_bv.setExternalStateVariable(material.mat_data.s1, "Temperature", material.temperature)
            # --- PRINT DATA
            print("----------------------------------------------------------------------------------------------------")
            print("+ TIME_STEP : {} | LOAD_VALUE : {}".format(time_step_index, time_step))
            outfile.write("----------------------------------------------------------------------------------------------------")
            outfile.write("\n")
            outfile.write("+ TIME_STEP : {} | LOAD_VALUE : {}".format(time_step_index, time_step))
            outfile.write("\n")
            iteration: int = 0
            break_iteration: bool = False
            while iteration < problem.number_of_iterations and not break_iteration:
                iteration_time_start = time.time()
                min_eigenvals: float = +np.inf
                max_eigenvals: float = -np.inf
                # --------------------------------------------------------------------------------------------------
                # SET SYSTEM MATRIX AND VECTOR
                # --------------------------------------------------------------------------------------------------
                tangent_matrix: ndarray = np.zeros((_constrained_system_size, _constrained_system_size), dtype=real)
                residual: ndarray = np.zeros((_constrained_system_size), dtype=real)
                # --------------------------------------------------------------------------------------------------
                # SET TIME INCREMENT
                # --------------------------------------------------------------------------------------------------
                if time_step_index == 0:
                    _dt: float = time_step
                else:
                    _dt: float = time_step - problem.time_steps[time_step_index - 1]
                # --------------------------------------------------------------------------------------------------
                # FOR ELEMENT LOOP
                # --------------------------------------------------------------------------------------------------
                iter_face_constraint: int = 0
                for boundary_condition in problem.boundary_conditions:
                    if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                        boundary_condition.force = 0.0
                # for _element_index, element in enumerate(problem.elements):
                #     _io: int = problem.finite_element.computation_integration_order
                #     cell_quadrature_size = element.cell.get_quadrature_size(
                #         # problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                #         _io, quadrature_type=problem.quadrature_type
                #     )
                #     for _qc in range(cell_quadrature_size):
                #         _qp = element.quad_p_indices[_qc]
                #         material.mat_data.s1.gradients[_qp] = transformation_gradient
                mean_cell_iterations: float = 0.0
                for _element_index, element in enumerate(problem.elements):
                    # print("processing cell : {}".format(_element_index))
                    # _io: int = problem.finite_element.k_order + problem.finite_element.k_order
                    _io: int = problem.finite_element.computation_integration_order
                    cell_quadrature_size = element.cell.get_quadrature_size(
                        # problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                        _io, quadrature_type=problem.quadrature_type
                    )
                    cell_quadrature_points = element.cell.get_quadrature_points(
                        _io, quadrature_type=problem.quadrature_type
                    )
                    cell_quadrature_weights = element.cell.get_quadrature_weights(
                        _io, quadrature_type=problem.quadrature_type
                    )
                    x_c: ndarray = element.cell.get_centroid()
                    bdc: ndarray = element.cell.get_bounding_box()
                    _nf: int = len(element.faces)
                    _c0_c: int = _dx * _cl
                    if solver_type == SolverType.CELL_EQUILIBRIUM:
                        # --- INITIALIZE CELL LOOP
                        local_iteration: int = 0
                        local_max_iteration: int = 10
                        local_tolerance: float = 1.e-6
                        cell_convergence: bool = False
                        local_loop_index: int = 0
                        local_split_max: int = 10
                        local_split_index: int = 0
                        local_faces_unknown_vectors = [np.copy(faces_unknown_vector)]
                        local_faces_unknown_vector_temp = np.copy(faces_unknown_vector)
                        while local_loop_index < len(local_faces_unknown_vectors) and not cell_convergence and local_split_index < local_split_max:
                            # print("_element_index : {}".format(_element_index))
                            # print("len(local_faces_unknown_vectors) : {}".format(len(local_faces_unknown_vectors)))
                            # print("local_loop_index : {}".format(local_loop_index))
                            # print("localiter : {}".format(local_iteration))
                            # print("cell_convergence : {}".format(cell_convergence))
                            break_local_loop: bool = False
                            # while local_iteration < local_max_iteration and not break_iteration and not break_local_iteration:
                            loc_faces_unknowns: ndarray = local_faces_unknown_vectors[local_loop_index]
                            while local_iteration < local_max_iteration and not break_local_loop and not cell_convergence:
                                # print("local_iteration : {}".format(local_iteration))
                                # print("break_local_loop : {}".format(break_local_loop))
                                _io: int = problem.finite_element.computation_integration_order
                                cell_quadrature_size = element.cell.get_quadrature_size(
                                    # problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                                    _io, quadrature_type=problem.quadrature_type
                                )
                                cell_quadrature_points = element.cell.get_quadrature_points(
                                    _io, quadrature_type=problem.quadrature_type
                                )
                                cell_quadrature_weights = element.cell.get_quadrature_weights(
                                    _io, quadrature_type=problem.quadrature_type
                                )
                                # --- INITIALIZE MATRIX AND VECTORS
                                element_stiffness_matrix = np.zeros((element.element_size, element.element_size),
                                                                    dtype=real)
                                element_internal_forces = np.zeros((element.element_size,), dtype=real)
                                element_external_forces = np.zeros((element.element_size,), dtype=real)
                                for _qc in range(cell_quadrature_size):
                                    _qp = element.quad_p_indices[_qc]
                                    _w_q_c = cell_quadrature_weights[_qc]
                                    _x_q_c = cell_quadrature_points[:, _qc]
                                    # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                                    transformation_gradient = element.get_transformation_gradient(loc_faces_unknowns, _qc)
                                    # transformation_gradient = element.get_transformation_gradient(faces_unknown_vector, _qc)
                                    material.mat_data.s1.gradients[_qp] = transformation_gradient
                                    # --- INTEGRATE BEHAVIOUR LAW
                                    # print("!!!! INTEGRATION START @ point : {}".format(_qp))
                                    integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp,
                                                                  (_qp + 1))
                                    # print("!!!! INTEGRATION STOP @ point : {}".format(_qp))
                                    if integ_res != 1:
                                        # print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {} WITH STRAIN {}".format(_element_index, _qp2, transformation_gradient))
                                        print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(
                                            _element_index, _qp))
                                        print("++++++++++++++++ - POINT {}".format(_x_q_c))
                                        print("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                                        outfile.write("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                                        outfile.write("\n")
                                        outfile.write("++++++++++++++++ - POINT {}".format(_x_q_c))
                                        outfile.write("\n")
                                        outfile.write("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                                        outfile.write("\n")
                                        # break_iteration = True
                                        break_local_loop = True
                                        break
                                    else:
                                        max_eigval = np.max(np.linalg.eigvals(material.mat_data.K[_qp]))
                                        min_eigval = np.min(np.linalg.eigvals(material.mat_data.K[_qp]))
                                        if max_eigval > max_eigenvals:
                                            max_eigenvals = max_eigval
                                        if min_eigval < min_eigenvals:
                                            min_eigenvals = min_eigval
                                        # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                        element_stiffness_matrix += _w_q_c * (
                                                element.gradients_operators[_qc].T @ material.mat_data.K[_qp] @
                                                element.gradients_operators[_qc]
                                        )
                                        # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                        element_internal_forces += _w_q_c * (
                                                element.gradients_operators[_qc].T @
                                                material.mat_data.s1.thermodynamic_forces[_qp]
                                        )
                                # if not break_iteration:
                                if not break_local_loop:
                                    # --- VOLUMETRIC LOAD
                                    _io: int = problem.finite_element.l_order
                                    _io: int = 8
                                    cell_quadrature_size = element.cell.get_quadrature_size(
                                        # problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                                        _io, quadrature_type=problem.quadrature_type
                                    )
                                    cell_quadrature_points = element.cell.get_quadrature_points(
                                        _io, quadrature_type=problem.quadrature_type
                                    )
                                    cell_quadrature_weights = element.cell.get_quadrature_weights(
                                        _io, quadrature_type=problem.quadrature_type
                                    )
                                    for _qc in range(cell_quadrature_size):
                                        _w_q_c = cell_quadrature_weights[_qc]
                                        _x_q_c = cell_quadrature_points[:, _qc]
                                        v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
                                        for load in problem.loads:
                                            vl = _w_q_c * v * load.function(time_step, _x_q_c)
                                            _re0 = load.direction * _cl
                                            _re1 = (load.direction + 1) * _cl
                                            element_external_forces[_re0:_re1] += vl
                                    # --- STAB PARAMETER CHANGE
                                    stab_param = material.stabilization_parameter
                                    # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                                    element_stiffness_matrix += stab_param * element.stabilization_operator
                                    # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                                    # element_internal_forces += (
                                    #         stab_param
                                    #         * element.stabilization_operator
                                    #         @ element.get_element_unknown_vector(faces_unknown_vector)
                                    # )
                                    element_internal_forces += (
                                            stab_param
                                            * element.stabilization_operator
                                            @ element.get_element_unknown_vector(loc_faces_unknowns)
                                    )
                                    K_cc = element_stiffness_matrix[:_c0_c, :_c0_c]
                                    local_element_residual = element_internal_forces - element_external_forces
                                    # print("@ CELL : {} : INT FORCE = {}".format(_element_index, element_internal_forces))
                                    # print("@ CELL : {} : EXT FORCE = {}".format(_element_index, element_external_forces))
                                    R_cc = local_element_residual[:_c0_c]
                                    local_external_forces_coefficient = np.max(np.abs(K_cc))
                                    if local_external_forces_coefficient == 0.0:
                                        local_external_forces_coefficient = 1.
                                    local_external_forces_coefficient = 1. * material.stabilization_parameter / np.prod(element.cell.get_bounding_box())
                                    # local_residual_evaluation = np.max(np.abs(R_cc) / local_external_forces_coefficient) * element.cell.get_diameter()
                                    local_residual_evaluation = np.max(np.abs(R_cc) / local_external_forces_coefficient)
                                    # print("LOCAL ITER : {} | RES MAX : {}".format(_local_iteration, local_residual_evaluation))
                                    if local_residual_evaluation < local_tolerance:
                                        # faces_unknown_vector_previous_step = np.copy(faces_unknown_vector)
                                        # for element in problem.elements:
                                        #     element.cell_unknown_vector_backup = np.copy(element.cell_unknown_vector)
                                        # time_step_index += 1
                                        mean_cell_iterations += float(local_iteration)
                                        # print("!!!!!!! THERE : {}".format(mean_cell_iterations))
                                        # break_local_iteration = True
                                        local_loop_index += 1
                                        cell_convergence = True
                                        local_iteration = 0
                                    elif local_iteration == local_max_iteration - 1:
                                        outfile.write(
                                            "++++++++++++++++ @ CELL : {} | MAX CONVERGENCE ACHIEVED WITH RES : {:.15E}".format(
                                                _element_index, local_residual_evaluation))
                                        outfile.write("\n")
                                        print("++++++++++++++++ @ CELL : {} | MAX CONVERGENCE ACHIEVED WITH RES : {:.15E}".format(
                                                _element_index,local_residual_evaluation))
                                        break_local_loop = True
                                    else:
                                        cell_correction = np.linalg.solve(-K_cc, R_cc)
                                        element.cell_unknown_vector += cell_correction
                                        # print("solving")
                                        local_iteration += 1
                                else:
                                    pass
                            if break_local_loop:
                                local_split_index += 1
                                # --- SPLITTING LOCAL UNKNOWN
                                print(
                                    "++++++++++++++++ @ CELL : {} | SPLITTING".format(
                                        _element_index))
                                local_faces_unknown_vectors.insert(local_loop_index,
                                                                   faces_unknown_vector_previous_step + local_faces_unknown_vector_temp / 2.0)
                                local_faces_unknown_vector_temp = local_faces_unknown_vectors[local_loop_index]
                                element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                                local_iteration = 0
                            else:
                                pass
                        if local_split_index == local_split_max:
                            break_iteration = True
                    elif solver_type == SolverType.STATIC_CONDENSATION:
                        # --- INITIALIZE MATRIX AND VECTORS
                        element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
                        element_internal_forces = np.zeros((element.element_size,), dtype=real)
                        element_external_forces = np.zeros((element.element_size,), dtype=real)
                        for _qc in range(cell_quadrature_size):
                            _qp = element.quad_p_indices[_qc]
                            _w_q_c = cell_quadrature_weights[_qc]
                            _x_q_c = cell_quadrature_points[:, _qc]
                            material.mat_data.K[0] = 4
                            material.mat_data.K[1] = 2
                            # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                            transformation_gradient = element.get_transformation_gradient(faces_unknown_vector, _qc)
                            material.mat_data.s1.gradients[_qp] = transformation_gradient
                            # --- INTEGRATE BEHAVIOUR LAW
                            integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
                            if integ_res != 1:
                                # print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {} WITH STRAIN {}".format(_element_index, _qp2, transformation_gradient))
                                print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                                print("++++++++++++++++ - POINT {}".format(_x_q_c))
                                print("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                                # print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {} WITH STRAIN {}".format(_element_index, _qp2, transformation_gradient))
                                outfile.write("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                                outfile.write("\n")
                                outfile.write("++++++++++++++++ - POINT {}".format(_x_q_c))
                                outfile.write("\n")
                                outfile.write("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                                outfile.write("\n")
                                break_iteration = True
                                break
                            else:
                                max_eigval = np.max(np.linalg.eigvals(material.mat_data.K[_qp]))
                                min_eigval = np.min(np.linalg.eigvals(material.mat_data.K[_qp]))
                                if max_eigval > max_eigenvals:
                                    max_eigenvals = max_eigval
                                if min_eigval < min_eigenvals:
                                    min_eigenvals = min_eigval
                                # if min_eigval <= 0.0:
                                #     break_iteration = True
                                # else:
                                # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                element_stiffness_matrix += _w_q_c * (
                                        element.gradients_operators[_qc].T @ material.mat_data.K[_qp] @
                                        element.gradients_operators[_qc]
                                )
                                # print("CEP : \n{}".format(material.mat_data.K[_qp]))
                                # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                element_internal_forces += _w_q_c * (
                                        element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
                                )
                                # v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
                                # for load in problem.loads:
                                #     vl = _w_q_c * v * load.function(time_step, _x_q_c)
                                #     _re0 = load.direction * _cl
                                #     _re1 = (load.direction + 1) * _cl
                                #     element_external_forces[_re0:_re1] += vl



                                # if min_eigval/material.stabilization_parameter < 1.0:

                                # print("int vars : {}".format(material.mat_data.s1.internal_state_variables[_qp]))
                                # print("gradients : {}".format(material.mat_data.s1.gradients[_qp]))
                                # print(
                                #     "min_eigval : {:.6E} | max_eigval : {:.6E} | stab : {:.6E}".format(min_eigval, max_eigval, material.stabilization_parameter))
                        if not break_iteration:
                            # --- VOLUMETRIC LOAD
                            _io: int = problem.finite_element.l_order
                            _io: int = 8
                            cell_quadrature_size = element.cell.get_quadrature_size(
                                # problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                                _io, quadrature_type=problem.quadrature_type
                            )
                            cell_quadrature_points = element.cell.get_quadrature_points(
                                _io, quadrature_type=problem.quadrature_type
                            )
                            cell_quadrature_weights = element.cell.get_quadrature_weights(
                                _io, quadrature_type=problem.quadrature_type
                            )
                            for _qc in range(cell_quadrature_size):
                                _w_q_c = cell_quadrature_weights[_qc]
                                _x_q_c = cell_quadrature_points[:, _qc]
                                v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
                                for load in problem.loads:
                                    vl = _w_q_c * v * load.function(time_step, _x_q_c)
                                    _re0 = load.direction * _cl
                                    _re1 = (load.direction + 1) * _cl
                                    element_external_forces[_re0:_re1] += vl
                            # --- STAB PARAMETER CHANGE
                            stab_param = material.stabilization_parameter
                            np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=None,suppress=True, threshold=sys.maxsize, formatter=None)
                            # print("int force : \n{}".format(element_internal_forces))
                            # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                            element_stiffness_matrix += stab_param * element.stabilization_operator
                            element_stiffness_matrix_prt = np.zeros(element_stiffness_matrix.shape, dtype=real)
                            for _row in range(element_stiffness_matrix.shape[0]):
                                for _col in range(element_stiffness_matrix.shape[1]):
                                    _row_n = element.get_diskpp_notation(problem.field, problem.finite_element, _row)
                                    _col_n = element.get_diskpp_notation(problem.field, problem.finite_element, _col)
                                    element_stiffness_matrix_prt[_row_n, _col_n] = element_stiffness_matrix[_row, _col]
                            # print("stiff mat : \n{}".format(repr(element_stiffness_matrix_prt)))
                            # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                            element_internal_forces += (
                                    stab_param
                                    * element.stabilization_operator
                                    @ element.get_element_unknown_vector(faces_unknown_vector)
                            )
                    if not break_iteration:
                        # --- BOUNDARY CONDITIONS
                        for boundary_condition in problem.boundary_conditions:
                            # --- DISPLACEMENT CONDITIONS
                            if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                                for f_local, f_global in enumerate(element.faces_indices):
                                    if f_global in problem.mesh.faces_boundaries_connectivity[
                                        boundary_condition.boundary_name]:
                                        # print("FLOC : {}, FGLOB : {}, ELEM : {}".format(f_local, f_global, _element_index))
                                        _l0 = _system_size + iter_face_constraint * _fk
                                        _l1 = _system_size + (iter_face_constraint + 1) * _fk
                                        _c0 = _cl * _dx + (f_local * _dx * _fk) + boundary_condition.direction * _fk
                                        _c1 = _cl * _dx + (f_local * _dx * _fk) + (
                                                    boundary_condition.direction + 1) * _fk
                                        _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
                                        _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
                                        face_lagrange = faces_unknown_vector[_l0:_l1]
                                        face_displacement = faces_unknown_vector[_r0:_r1]
                                        _m_psi_psi_face = np.zeros((_fk, _fk), dtype=real)
                                        _v_face_imposed_displacement = np.zeros((_fk,), dtype=real)
                                        face = element.faces[f_local]
                                        x_f = face.get_centroid()
                                        face_rot = face.get_rotation_matrix()
                                        force_item = 0.0
                                        bdf_proj = face.get_face_bounding_box()
                                        _io: int = problem.finite_element.k_order + problem.finite_element.k_order
                                        _io: int = 8
                                        face_quadrature_size = face.get_quadrature_size(
                                            _io,
                                            quadrature_type=problem.quadrature_type,
                                        )
                                        face_quadrature_points = face.get_quadrature_points(
                                            _io,
                                            quadrature_type=problem.quadrature_type,
                                        )
                                        face_quadrature_weights = face.get_quadrature_weights(
                                            _io,
                                            quadrature_type=problem.quadrature_type,
                                        )
                                        for _qf in range(face_quadrature_size):
                                            _x_q_f = face_quadrature_points[:, _qf]
                                            _w_q_f = face_quadrature_weights[_qf]
                                            _s_q_f = (face_rot @ _x_q_f)[:-1]
                                            _s_f = (face_rot @ x_f)[:-1]
                                            # v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                            #                                                           bdf_proj)
                                            # _v_face_imposed_displacement += (
                                            #         _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                            # )
                                            # force_item += (
                                            #     # _w_q_f * v @ face_lagrange / bdf_proj[0]
                                            #     #     _w_q_f * v @ face_lagrange
                                            #         _w_q_f * v[0] * face_lagrange[0]
                                            #     # v @ face_lagrange
                                            #     # face_lagrange[0]
                                            # )
                                            _psi_k = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                                                                                           bdf_proj)
                                            _m_psi_psi_face += _w_q_f * np.tensordot(_psi_k, _psi_k, axes=0)
                                        # _io: int = problem.finite_element.k_order
                                        _io: int = 8
                                        face_quadrature_size = face.get_quadrature_size(
                                            _io, quadrature_type=problem.quadrature_type,
                                        )
                                        face_quadrature_points = face.get_quadrature_points(
                                            _io, quadrature_type=problem.quadrature_type,
                                        )
                                        face_quadrature_weights = face.get_quadrature_weights(
                                            _io, quadrature_type=problem.quadrature_type,
                                        )
                                        for _qf in range(face_quadrature_size):
                                            _x_q_f = face_quadrature_points[:, _qf]
                                            _w_q_f = face_quadrature_weights[_qf]
                                            _s_q_f = (face_rot @ _x_q_f)[:-1]
                                            _s_f = (face_rot @ x_f)[:-1]
                                            v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                                                                                      bdf_proj)
                                            _v_face_imposed_displacement += (
                                                    _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                            )
                                            force_item += (
                                                # _w_q_f * v @ face_lagrange / bdf_proj[0]
                                                #     _w_q_f * v @ face_lagrange
                                                    _w_q_f * v[0] * face_lagrange[0]
                                                # v @ face_lagrange
                                                # face_lagrange[0]
                                            )
                                            # _psi_k = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                            #                                                                bdf_proj)
                                            # _m_psi_psi_face += _w_q_f * np.tensordot(_psi_k, _psi_k, axes=0)
                                        force_item *= material.lagrange_parameter / face.get_diameter()
                                        boundary_condition.force += force_item
                                        # if f_global == 0:
                                        #     boundary_condition.force = material.lagrange_parameter * face_lagrange[0]
                                        _m_psi_psi_face_inv = np.linalg.inv(_m_psi_psi_face)
                                        imposed_face_displacement = _m_psi_psi_face_inv @ _v_face_imposed_displacement
                                        # imposed_face_displacement = np.zeros((_fk,))
                                        # imposed_face_displacement[0] = boundary_condition.function(time_step, 0.0)
                                        face_displacement_difference = face_displacement - imposed_face_displacement
                                        # print("face lag @ face : {} is --> \n{}".format(f_global, list(imposed_face_displacement)))
                                        # --- LAGRANGE INTERNAL FORCES PART
                                        element_internal_forces[
                                        _c0:_c1] += material.lagrange_parameter * face_lagrange
                                        residual[
                                        _l0:_l1] -= material.lagrange_parameter * face_displacement_difference
                                        # --- LAGRANGE MATRIX PART
                                        tangent_matrix[_l0:_l1, _r0:_r1] += material.lagrange_parameter * np.eye(
                                            _fk, dtype=real
                                        )
                                        tangent_matrix[_r0:_r1, _l0:_l1] += material.lagrange_parameter * np.eye(
                                            _fk, dtype=real
                                        )
                                        # --- SET EXTERNAL FORCES COEFFICIENT
                                        lagrange_external_forces = (
                                                material.lagrange_parameter * imposed_face_displacement
                                        )
                                        if np.max(np.abs(lagrange_external_forces)) > external_forces_coefficient:
                                            external_forces_coefficient = np.max(np.abs(lagrange_external_forces))
                                        iter_face_constraint += 1
                            elif boundary_condition.boundary_type == BoundaryType.PRESSURE:
                                for f_local, f_global in enumerate(element.faces_indices):
                                    if f_global in problem.mesh.faces_boundaries_connectivity[
                                        boundary_condition.boundary_name]:
                                        face = element.faces[f_local]
                                        x_f = face.get_centroid()
                                        face_rot = face.get_rotation_matrix()
                                        _io: int = problem.finite_element.k_order
                                        _io: int = 8
                                        face_quadrature_size = face.get_quadrature_size(
                                            _io, quadrature_type=problem.quadrature_type,
                                        )
                                        face_quadrature_points = face.get_quadrature_points(
                                            _io, quadrature_type=problem.quadrature_type,
                                        )
                                        face_quadrature_weights = face.get_quadrature_weights(
                                            _io, quadrature_type=problem.quadrature_type,
                                        )
                                        for _qf in range(face_quadrature_size):
                                            _x_q_f = face_quadrature_points[:, _qf]
                                            _w_q_f = face_quadrature_weights[_qf]
                                            _s_q_f = (face_rot @ _x_q_f)[:-1]
                                            _s_f = (face_rot @ x_f)[:-1]
                                            bdf_proj = face.get_face_bounding_box()
                                            v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                                                                                      bdf_proj)
                                            vf = _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                            _c0 = _dx * _cl + f_local * _dx * _fk + boundary_condition.direction * _fk
                                            _c1 = _dx * _cl + f_local * _dx * _fk + (
                                                        boundary_condition.direction + 1) * _fk
                                            element_external_forces[_c0:_c1] += vf
                        # print("cell done")
                        # --- COMPUTE RESIDUAL AFTER VOLUMETRIC CONTRIBUTION
                        element_residual = element_internal_forces - element_external_forces
                        # --------------------------------------------------------------------------------------------------
                        # CONDENSATION
                        # --------------------------------------------------------------------------------------------------
                        m_cell_cell = element_stiffness_matrix[:_c0_c, :_c0_c]
                        m_cell_faces = element_stiffness_matrix[:_c0_c, _c0_c:]
                        m_faces_cell = element_stiffness_matrix[_c0_c:, :_c0_c]
                        m_faces_faces = element_stiffness_matrix[_c0_c:, _c0_c:]
                        v_cell = -element_residual[:_c0_c]
                        v_faces = -element_residual[_c0_c:]
                        _condtest = np.linalg.cond(m_cell_cell)
                        if _condtest > 1.e10:
                            print("++++++++++++++++ @ CELL : {} | CONDITIONING : {}".format(_element_index, _condtest))
                            outfile.write("++++++++++++++++ @ CELL : {} | CONDITIONING : {}".format(_element_index, _condtest))
                            outfile.write("\n")
                        m_cell_cell_inv = np.linalg.inv(m_cell_cell)
                        K_cond = m_faces_faces - ((m_faces_cell @ m_cell_cell_inv) @ m_cell_faces)
                        R_cond = v_faces - (m_faces_cell @ m_cell_cell_inv) @ v_cell
                        # --- SET CONDENSATION/DECONDENSATION MATRICES
                        element.m_cell_cell_inv = m_cell_cell_inv
                        element.m_cell_faces = m_cell_faces
                        element.v_cell = v_cell
                        # --- ASSEMBLY
                        for _i_local, _i_global in enumerate(element.faces_indices):
                            _rg0 = _i_global * (_fk * _dx)
                            _rg1 = (_i_global + 1) * (_fk * _dx)
                            _re0 = _i_local * (_fk * _dx)
                            _re1 = (_i_local + 1) * (_fk * _dx)
                            residual[_rg0:_rg1] += R_cond[_re0:_re1]
                            for _j_local, _j_global in enumerate(element.faces_indices):
                                _cg0 = _j_global * (_fk * _dx)
                                _cg1 = (_j_global + 1) * (_fk * _dx)
                                _ce0 = _j_local * (_fk * _dx)
                                _ce1 = (_j_local + 1) * (_fk * _dx)
                                tangent_matrix[_rg0:_rg1, _cg0:_cg1] += K_cond[_re0:_re1, _ce0:_ce1]
                        # --- SET EXTERNAL FORCES COEFFICIENT
                        if np.max(np.abs(element_external_forces)) > external_forces_coefficient:
                            external_forces_coefficient = np.max(np.abs(element_external_forces))
                    else:
                        break
                if not break_iteration:
                    print("++++ DONE ITERATION OVER ELEMENTS")
                    print("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E} | STAB : {:.6E}".format(min_eigenvals, max_eigenvals, material.stabilization_parameter))
                    # print("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E}".format(min_eigenvals/material.stabilization_parameter, max_eigenvals/material.stabilization_parameter))
                    outfile.write("++++ DONE ITERATION OVER ELEMENTS")
                    outfile.write("\n")
                    outfile.write("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E} | STAB : {:.6E}".format(min_eigenvals, max_eigenvals, material.stabilization_parameter))
                    outfile.write("\n")
                    # outfile.write("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E}".format(min_eigenvals/material.stabilization_parameter, max_eigenvals/material.stabilization_parameter))
                    # outfile.write("\n")
                    # --------------------------------------------------------------------------------------------------
                    # MEAN NUMBER OF LOCAL ITERATIONS
                    # --------------------------------------------------------------------------------------------------
                    mean_cell_iterations /= float(len(problem.elements))
                    if solver_type == SolverType.CELL_EQUILIBRIUM:
                        print("++++ ITER : {} | MEAN CELL ITERATIONS : {:.6E}".format(str(iteration).zfill(4), mean_cell_iterations))
                        outfile.write("++++ ITER : {} | MEAN CELL ITERATIONS : {:.6E}".format(str(iteration).zfill(4), mean_cell_iterations))
                        outfile.write("\n")
                    # --------------------------------------------------------------------------------------------------
                    # RESIDUAL EVALUATION
                    # --------------------------------------------------------------------------------------------------
                    if external_forces_coefficient == 0.0:
                        external_forces_coefficient = 1.0
                    print("++++ RESIDUAL EVALUATION")
                    outfile.write("++++ RESIDUAL EVALUATION")
                    outfile.write("\n")
                    residual_evaluation = np.max(np.abs(residual)) / external_forces_coefficient
                    if residual_evaluation < problem.tolerance:
                        print(
                            "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                        outfile.write(
                            "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                        outfile.write("\n")
                        # --- UPDATE INTERNAL VARIABLES
                        mgis_bv.update(material.mat_data)
                        file_suffix = "{}".format(time_step_index).zfill(6)
                        for bc in problem.boundary_conditions:
                            if bc.boundary_type == BoundaryType.DISPLACEMENT:
                                bc.force_values.append(bc.force)
                                bc.time_values.append(time_step)
                        problem.create_vertex_res_files(problem.res_folder_path, file_suffix)
                        problem.create_quadrature_points_res_files(problem.res_folder_path, file_suffix, material)
                        problem.write_vertex_res_files(problem.res_folder_path, file_suffix, faces_unknown_vector)
                        problem.write_quadrature_points_res_files(problem.res_folder_path, file_suffix, material,
                                                                  faces_unknown_vector)
                        problem.fill_quadrature_internal_variables_output(
                            problem.res_folder_path, "INTERNAL_VARIABLES", time_step_index, time_step, material
                        )
                        # problem.create_output(problem.res_folder_path)
                        problem.fill_quadrature_stress_output(problem.res_folder_path, "CAUCHY_STRESS",
                                                              time_step_index, time_step, material)
                        problem.fill_quadrature_strain_output(problem.res_folder_path, "STRAIN", time_step_index,
                                                              time_step, material)
                        problem.fill_quadrature_displacement_output(problem.res_folder_path, "QUADRATURE_DISPLACEMENT",
                                                                    time_step_index, time_step, faces_unknown_vector)
                        problem.fill_node_displacement_output(problem.res_folder_path, "NODE_DISPLACEMENT",
                                                              time_step_index, time_step, faces_unknown_vector)
                        for bc in problem.boundary_conditions:
                            problem.write_force_output(problem.res_folder_path, bc)
                        faces_unknown_vector_previous_step = np.copy(faces_unknown_vector)
                        for element in problem.elements:
                            element.cell_unknown_vector_backup = np.copy(element.cell_unknown_vector)
                        step_time_stop = time.time()
                        step_elapsed_time = step_time_stop - step_time_start
                        iteration_elapsed_time = step_time_stop - iteration_time_start
                        print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        outfile.write("\n")
                        print("+ ITERATIONS : {}".format(iteration + 1))
                        outfile.write("+ ITERATIONS : {}".format(iteration + 1))
                        outfile.write("\n")
                        print("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        outfile.write("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        outfile.write("\n")
                        iteration = 0
                        time_step_temp = time_step + 0.
                        time_step_index += 1
                        break_iteration = True
                    elif iteration == problem.number_of_iterations - 1:
                        print("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                            str(iteration).zfill(4), residual_evaluation))
                        outfile.write("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                            str(iteration).zfill(4), residual_evaluation))
                        outfile.write("\n")
                        faces_unknown_vector = np.copy(faces_unknown_vector_previous_step)
                        for element in problem.elements:
                            element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                        iteration = 0
                        problem.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                        step_time_stop = time.time()
                        step_elapsed_time = step_time_stop - step_time_start
                        iteration_elapsed_time = step_time_stop - iteration_time_start
                        print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        outfile.write("\n")
                        print("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        outfile.write("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        outfile.write("\n")
                        break_iteration = True
                    else:
                        # --- SOLVE SYSTEM
                        # print("++++ ITER : {} | RES_MAX : {:.6E} | COND : {:.6E}".format(
                        #     str(iteration).zfill(4), residual_evaluation,
                        #     np.linalg.cond(tangent_matrix)))
                        print("++++ ITER : {} | RES_MAX : {:.6E}".format(
                            str(iteration).zfill(4), residual_evaluation))
                        outfile.write("++++ ITER : {} | RES_MAX : {:.6E}".format(
                            str(iteration).zfill(4), residual_evaluation))
                        outfile.write("\n")
                        sparse_global_matrix = csr_matrix(tangent_matrix)
                        print("++++++++ SOLVING THE SYSTEM")
                        outfile.write("++++++++ SOLVING THE SYSTEM")
                        outfile.write("\n")
                        scaling_factor = np.max(np.abs(tangent_matrix))
                        solving_start_time = time.time()
                        correction = spsolve(sparse_global_matrix / scaling_factor, residual / scaling_factor)
                        solving_end_time = time.time()
                        system_check = np.max(np.abs(tangent_matrix @ correction - residual))
                        print("++++++++ SYSTEM SOLVED IN : {:.6E}s | SYSTEM CHECK : {:.6E}".format(
                            solving_end_time - solving_start_time, system_check))
                        outfile.write("++++++++ SYSTEM SOLVED IN : {:.6E}s | SYSTEM CHECK : {:.6E}".format(
                            solving_end_time - solving_start_time, system_check))
                        outfile.write("\n")
                        faces_unknown_vector += correction
                        # --- DECONDENSATION
                        # if solver_type == SolverType.STATIC_CONDENSATION or solver_type == SolverType.CELL_EQUILIBRIUM:
                        if solver_type == SolverType.STATIC_CONDENSATION:
                            print("----------------- DECONDENSSSSSS".format(iteration_elapsed_time))
                            for element in problem.elements:
                                _nf = len(element.faces)
                                face_correction = np.zeros((_nf * _fk * _dx), dtype=real)
                                for _i_local, _i_global in enumerate(element.faces_indices):
                                    _c0_fg = _i_global * (_fk * _dx)
                                    _c1_fg = (_i_global + 1) * (_fk * _dx)
                                    _c0_fl = _i_local * (_fk * _dx)
                                    _c1_fl = (_i_local + 1) * (_fk * _dx)
                                    face_correction[_c0_fl:_c1_fl] += correction[_c0_fg:_c1_fg]
                                cell_correction = element.m_cell_cell_inv @ (
                                        element.v_cell - element.m_cell_faces @ face_correction
                                )
                                # --- ADDING CORRECTION TO CURRENT DISPLACEMENT
                                element.cell_unknown_vector += cell_correction
                        iteration_time_stop = time.time()
                        iteration_elapsed_time = iteration_time_stop - iteration_time_start
                        print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        outfile.write("\n")
                        iteration += 1
                else:
                    print("+ SPLITTING TIME STEP")
                    outfile.write("SPLITTING TIME STEP\n")
                    problem.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                    faces_unknown_vector = np.copy(faces_unknown_vector_previous_step)
                    for element in problem.elements:
                        element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                    iteration = 0
                    step_time_stop = time.time()
                    iteration_elapsed_time = step_time_stop - iteration_time_start
                    print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                    outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                    outfile.write("\n")
                    step_elapsed_time = step_time_stop - step_time_start
                    print("+ TIME_STEP_TIME : {:.6E}".format(step_elapsed_time))
                    outfile.write("+ TIME_STEP_TIME : {:.6E}".format(step_elapsed_time))
                    outfile.write("\n")
        computation_time_stop = time.time()
        compuattion_elapsed_time = computation_time_stop - computation_time_start
        print("----------------------------------------------------------------------------------------------------")
        print("COMPUTATION_TIME : {:.6E}".format(compuattion_elapsed_time))
        outfile.write("----------------------------------------------------------------------------------------------------")
        outfile.write("\n")
        outfile.write("COMPUTATION_TIME : {:.6E}".format(compuattion_elapsed_time))
        outfile.write("\n")
