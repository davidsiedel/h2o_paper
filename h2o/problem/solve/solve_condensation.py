import time

import numpy as np

import tfel
import tfel.math

from mgis.behaviour import IntegrationType

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

def decond(cell):
    return 0

def solve_condensation(problem: Problem, material: Material, verbose: bool = False, debug_mode: DebugMode = DebugMode.NONE):
    clean_res_dir(problem.res_folder_path)
    problem.create_output(problem.res_folder_path)
    output_file_path = os.path.join(problem.res_folder_path, "output.txt")
    computation_tic = time.time()
    # MIN
    min_global_time_step_time: float = np.inf
    min_global_iteration_time: float = np.inf
    # MAX
    max_global_time_step_time: float = -np.inf
    max_global_iteration_time: float = -np.inf
    # MEAN
    mean_global_time_step_time: float = 0.0
    mean_global_iteration_time: float = 0.0
    # --- SUM
    sum_global_time_step_time: float = 0.0
    sum_global_iteration_time: float = 0.0
    # COUNT
    cnt_global_time_step: int = 0
    cnt_global_iteration: int = 0
    # 
    # 
    # 
    num_total_skeleton_iterations = 0
    num_total_skeleton_time_steps = 0
    with open(output_file_path, "a") as outfile:

        def write_out_msg(msg: str):
            outfile.write(msg)
            outfile.write("\n")

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
        outfile.write(
            "----------------------------------------------------------------------------------------------------\n")
        outfile.write("****************** COMPUTATION\n")
        outfile.write("+ SYSTEM SIZE : {}\n".format(_system_size))
        print("+ SYSTEM SIZE : {}".format(_system_size))
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
        while time_step_index < len(problem.time_steps):
            global_time_step_tic = time.time()
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
            # --------------------------------------------------------------------------------------------------
            # INIT ANDERSON ACCELERATION
            # --------------------------------------------------------------------------------------------------
            nit = 10
            freq = 1
            acceleration_u = tfel.math.UAnderson(nit, freq)
            acceleration_u.initialize(faces_unknown_vector)
            while iteration < problem.number_of_iterations and not break_iteration:
                global_iteration_tic = time.time()
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
                for _element_index, element in enumerate(problem.elements):
                    _io: int = problem.finite_element.computation_integration_order
                    cell_quadrature_size = element.cell.get_quadrature_size(
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
                    # --- INITIALIZE MATRIX AND VECTORS
                    element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
                    element_internal_forces = np.zeros((element.element_size,), dtype=real)
                    element_external_forces = np.zeros((element.element_size,), dtype=real)
                    for _qc in range(cell_quadrature_size):
                        _qp = element.quad_p_indices[_qc]
                        _w_q_c = cell_quadrature_weights[_qc]
                        _x_q_c = cell_quadrature_points[:, _qc]
                        # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                        transformation_gradient = element.get_transformation_gradient(faces_unknown_vector, _qc)
                        material.mat_data.s1.gradients[_qp] = transformation_gradient
                        # --- INTEGRATE BEHAVIOUR LAW
                        integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
                        # integration_type = IntegrationType.IntegrationWithElasticOperator
                        # # integration_type = IntegrationType.IntegrationWithConsistentTangentOperator
                        # integ_res = mgis_bv.integrate(material.mat_data, integration_type, _dt, _qp, (_qp + 1))
                        if integ_res != 1:
                            # print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {} WITH STRAIN {}".format(_element_index, _qp2, transformation_gradient))
                            print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                            print("++++++++++++++++ - POINT {}".format(_x_q_c))
                            print("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                            write_out_msg("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                            write_out_msg("++++++++++++++++ - POINT {}".format(_x_q_c))
                            write_out_msg("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                            # print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {} WITH STRAIN {}".format(_element_index, _qp2, transformation_gradient))
                            # outfile.write("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                            # outfile.write("\n")
                            # outfile.write("++++++++++++++++ - POINT {}".format(_x_q_c))
                            # outfile.write("\n")
                            # outfile.write("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                            # outfile.write("\n")
                            break_iteration = True
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
                            # print("CEP : \n{}".format(material.mat_data.K[_qp]))
                            # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                            element_internal_forces += _w_q_c * (
                                    element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
                            )
                    if not break_iteration:
                        # --- VOLUMETRIC LOAD
                        _io: int = problem.finite_element.l_order
                        _io: int = problem.finite_element.computation_integration_order
                        cell_quadrature_size = element.cell.get_quadrature_size(
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
                        # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                        element_stiffness_matrix += stab_param * element.stabilization_operator
                        # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                        element_internal_forces += (
                                stab_param
                                * element.stabilization_operator
                                @ element.get_element_unknown_vector(faces_unknown_vector)
                        )
                        # --- BOUNDARY CONDITIONS
                        for boundary_condition in problem.boundary_conditions:
                            # --- DISPLACEMENT CONDITIONS
                            if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                                for f_local, f_global in enumerate(element.faces_indices):
                                    if f_global in problem.mesh.faces_boundaries_connectivity[
                                        boundary_condition.boundary_name]:
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
                                        _io: int = problem.finite_element.computation_integration_order
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
                                            _psi_k = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                                                                                           bdf_proj)
                                            _m_psi_psi_face += _w_q_f * np.tensordot(_psi_k, _psi_k, axes=0)
                                        _io: int = 8
                                        _io: int = problem.finite_element.computation_integration_order
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
                                        force_item = material.lagrange_parameter * (np.ones(_fk) @ face_lagrange[:])
                                        boundary_condition.force += force_item
                                        _m_psi_psi_face_inv = np.linalg.inv(_m_psi_psi_face)
                                        imposed_face_displacement = _m_psi_psi_face_inv @ _v_face_imposed_displacement
                                        face_displacement_difference = face_displacement - imposed_face_displacement
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
                                        _io: int = problem.finite_element.computation_integration_order
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
                    # print("++++ DONE ITERATION OVER ELEMENTS")
                    # print("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E} | STAB : {:.6E}".format(min_eigenvals, max_eigenvals, material.stabilization_parameter))
                    # print("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E}".format(min_eigenvals/material.stabilization_parameter, max_eigenvals/material.stabilization_parameter))
                    # outfile.write("++++ DONE ITERATION OVER ELEMENTS")
                    # outfile.write("\n")
                    # outfile.write("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E} | STAB : {:.6E}".format(min_eigenvals, max_eigenvals, material.stabilization_parameter))
                    # outfile.write("\n")
                    # outfile.write("++++ MIN EIGVALS : {:.6E} | MAX EIGVALS : {:.6E}".format(min_eigenvals/material.stabilization_parameter, max_eigenvals/material.stabilization_parameter))
                    # outfile.write("\n")
                    # --------------------------------------------------------------------------------------------------
                    # RESIDUAL EVALUATION
                    # --------------------------------------------------------------------------------------------------
                    if external_forces_coefficient == 0.0:
                        external_forces_coefficient = 1.0
                    # print("++++ RESIDUAL EVALUATION")
                    # outfile.write("++++ RESIDUAL EVALUATION")
                    # outfile.write("\n")
                    residual_evaluation = np.max(np.abs(residual)) / external_forces_coefficient
                    if residual_evaluation < problem.tolerance:
                        print(
                            "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                        write_out_msg(
                            "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                        # outfile.write(
                        #     "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                        #         str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                        # outfile.write("\n")
                        # --- UPDATE INTERNAL VARIABLES
                        mgis_bv.update(material.mat_data)
                        file_suffix = "{}".format(time_step_index).zfill(6)
                        for bc in problem.boundary_conditions:
                            if bc.boundary_type == BoundaryType.DISPLACEMENT:
                                bc.force_values.append(bc.force)
                                bc.time_values.append(time_step)
                        # problem.create_vertex_res_files(problem.res_folder_path, file_suffix)
                        # problem.create_quadrature_points_res_files(problem.res_folder_path, file_suffix, material)
                        # problem.write_vertex_res_files(problem.res_folder_path, file_suffix, faces_unknown_vector)
                        # problem.write_quadrature_points_res_files(problem.res_folder_path, file_suffix, material,
                        #                                           faces_unknown_vector)
                        if not material.behaviour_name in ["Elasticity", "Signorini"]:
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
                        # print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        # outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        # outfile.write("\n")
                        print("+ ITERATIONS : {}".format(iteration + 1))
                        write_out_msg("+ ITERATIONS : {}".format(iteration + 1))
                        # print("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        # outfile.write("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        # outfile.write("\n")
                        iteration = 0
                        time_step_temp = time_step + 0.
                        time_step_index += 1
                        num_total_skeleton_time_steps += 1
                        # TIME STEP CORRECTION

                        break_iteration = True
                    elif iteration == problem.number_of_iterations - 1:
                        print("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                            str(iteration).zfill(4), residual_evaluation))
                        write_out_msg("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                            str(iteration).zfill(4), residual_evaluation))
                        faces_unknown_vector = np.copy(faces_unknown_vector_previous_step)
                        for element in problem.elements:
                            element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                        iteration = 0
                        problem.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                        step_time_stop = time.time()
                        step_elapsed_time = step_time_stop - step_time_start
                        iteration_elapsed_time = step_time_stop - iteration_time_start
                        # print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        # outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        # outfile.write("\n")
                        # print("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        # outfile.write("+ TIME_STEP_TIME : {:.6E} s".format(step_elapsed_time))
                        # outfile.write("\n")
                        break_iteration = True
                    else:
                        # --- SOLVE SYSTEM
                        print("++++ ITER : {} | RES_MAX : {:.6E}".format(
                            str(iteration).zfill(4), residual_evaluation))
                        write_out_msg("++++ ITER : {} | RES_MAX : {:.6E}".format(
                            str(iteration).zfill(4), residual_evaluation))
                        sparse_global_matrix = csr_matrix(tangent_matrix)
                        # print("++++++++ SOLVING THE SYSTEM")
                        # outfile.write("++++++++ SOLVING THE SYSTEM")
                        # outfile.write("\n")
                        scaling_factor = np.max(np.abs(tangent_matrix))
                        solving_start_time = time.time()
                        correction = spsolve(sparse_global_matrix / scaling_factor, residual / scaling_factor)
                        solving_end_time = time.time()
                        system_check = np.max(np.abs(tangent_matrix @ correction - residual))
                        # print("++++++++ SYSTEM SOLVED IN : {:.6E}s | SYSTEM CHECK : {:.6E}".format(
                        #     solving_end_time - solving_start_time, system_check))
                        # outfile.write("++++++++ SYSTEM SOLVED IN : {:.6E}s | SYSTEM CHECK : {:.6E}".format(
                        #     solving_end_time - solving_start_time, system_check))
                        # outfile.write("\n")
                        faces_unknown_vector += correction
                        # --- DECONDENSATION
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
                        # print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        # outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                        # outfile.write("\n")
                        iteration += 1
                        # --------------------------------------------------------------------------------------------------
                        # ANDERSON ACCELERATE
                        # --------------------------------------------------------------------------------------------------
                        # acceleration_u.accelerate(faces_unknown_vector)
                        num_total_skeleton_iterations += 1
                else:
                    print("+ SPLITTING TIME STEP")
                    write_out_msg("+ SPLITTING TIME STEP")
                    problem.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                    faces_unknown_vector = np.copy(faces_unknown_vector_previous_step)
                    for element in problem.elements:
                        element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                    iteration = 0
                    step_time_stop = time.time()
                    iteration_elapsed_time = step_time_stop - iteration_time_start
                    # print("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                    # outfile.write("++++ ITER_TIME : {:.6E} s".format(iteration_elapsed_time))
                    # outfile.write("\n")
                    # step_elapsed_time = step_time_stop - step_time_start
                    # print("+ TIME_STEP_TIME : {:.6E}".format(step_elapsed_time))
                    # outfile.write("+ TIME_STEP_TIME : {:.6E}".format(step_elapsed_time))
                    # outfile.write("\n")
                # --- END OF ITERATION
                cnt_global_iteration += 1
                global_iteration_toc = time.time()
                global_iteration_time = global_iteration_toc - global_iteration_tic
                if global_iteration_time < min_global_iteration_time:
                    min_global_iteration_time = global_iteration_time
                if global_iteration_time > max_global_iteration_time:
                    max_global_iteration_time = global_iteration_time
                mean_global_iteration_time += global_iteration_time
                # write_out_msg("++++ ITER : {} | GLOBAL_ITERATION_TIME : {:.6E}".format(str(iteration).zfill(4),
                #                                                                        global_iteration_time))
            # --- END OF TIME STEP
            mean_global_iteration_time /= float(cnt_global_iteration)
            cnt_global_time_step += 1
            global_time_step_toc = time.time()
            global_time_step_time = global_time_step_toc - global_time_step_tic
            if global_time_step_time < min_global_time_step_time:
                min_global_time_step_time = global_time_step_time
            if global_time_step_time > max_global_time_step_time:
                max_global_time_step_time = global_time_step_time
            # write_out_msg(
            #     "+ TIME_STEP : {} | NUM_ITERATIONS : {}".format(str(time_step_index).zfill(4), cnt_global_iteration))
            # write_out_msg("+ TIME_STEP : {} | MEAN_ITERATION_TIME : {:.6E}".format(str(time_step_index).zfill(4),
            #                                                                        mean_global_iteration_time))
            # write_out_msg("+ TIME_STEP : {} | GLOBAL_TIME_STEP_TIME : {:.6E}".format(str(time_step_index).zfill(4),
            #                                                                          global_time_step_time))
            cnt_global_iteration = 0
            mean_global_iteration_time = 0.0
            mean_global_time_step_time += global_time_step_time
        # --- END OF COMPUTATION
        mean_global_time_step_time /= float(cnt_global_time_step)
        computation_toc = time.time()
        computation_time = computation_toc - computation_tic
        # write_out_msg("+ COMPUTATION | NUM_TIME_STEPS : {}".format(cnt_global_time_step))
        # write_out_msg("+ COMPUTATION | MEAN_TIME_STEP_TIME : {:.6E}".format(mean_global_time_step_time))
        # write_out_msg("+ COMPUTATION GLOBAL TIME : {:.6E}".format(computation_time))
        write_out_msg("+ COMPUTATION NUM TIME STEPS : {:.6E}".format(num_total_skeleton_time_steps))
        write_out_msg("+ COMPUTATION NUM ITERATIONS : {:.6E}".format(num_total_skeleton_iterations))
