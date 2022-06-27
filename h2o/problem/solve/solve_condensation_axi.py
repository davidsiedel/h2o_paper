import time
from tracemalloc import stop

import tfel
import tfel.math

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


def solve_condensation(
    problem: Problem,
    material: Material,
    verbose: bool = False,
    debug_mode: DebugMode = DebugMode.NONE,
    accelerate:int = 0,
    num_local_iterations: int = 200
):
    clean_res_dir(problem.res_folder_path)
    problem.create_output(problem.res_folder_path)
    output_file_path = os.path.join(problem.res_folder_path, "output.txt")
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
        _cell_system_size = problem.get_cell_system_size()
        _total_system_size = _cell_system_size + _constrained_system_size
        total_unknown_vector: ndarray = np.zeros((_total_system_size), dtype=real)
        total_unknown_vector_previous_step: ndarray = np.zeros((_total_system_size), dtype=real)
        # --- TIME STEP INIT
        time_step_index: int = 0
        time_step_temp: float = problem.time_steps[0]
        while time_step_index < len(problem.time_steps):
            time_step: float = problem.time_steps[time_step_index]
            material.set_temperature()
            # --- PRINT DATA
            print("----------------------------------------------------------------------------------------------------")
            print("+ TIME_STEP : {} | LOAD_VALUE : {}".format(time_step_index, time_step))
            write_out_msg("----------------------------------------------------------------------------------------------------")
            write_out_msg("+ TIME_STEP : {} | LOAD_VALUE : {}".format(time_step_index, time_step))
            iteration: int = 0
            break_iteration: bool = False
            # --------------------------------------------------------------------------------------------------
            # INIT ANDERSON ACCELERATION
            # --------------------------------------------------------------------------------------------------
            acceleration_u = tfel.math.UAnderson(3, 1)
            if accelerate > 0 and num_local_iterations > 0:
                acceleration_u.initialize(total_unknown_vector[:_constrained_system_size])
            elif accelerate > 0 and num_local_iterations == 0:
                acceleration_u.initialize(total_unknown_vector)
            # --------------------------------------------------------------------------------------------------
            # ITERATIONS
            # --------------------------------------------------------------------------------------------------
            while iteration < problem.number_of_iterations and not break_iteration:
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
                num_cells_iterations = 0
                for boundary_condition in problem.boundary_conditions:
                    if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                        boundary_condition.force = 0.0
                for _element_index, element in enumerate(problem.elements):

                    _nf: int = len(element.faces)
                    _c0_c: int = _dx * _cl
                    bdc: ndarray = element.cell.get_bounding_box()
                    
                    def make_cell_procedure(break_iter: bool):
                        # --------------------------------------------------------------------------------------------------
                        # INTEGRATION
                        # --------------------------------------------------------------------------------------------------
                        _io: int = problem.finite_element.computation_integration_order
                        cell_quadrature_size = element.cell.get_quadrature_size(_io, quadrature_type=problem.quadrature_type)
                        cell_quadrature_points = element.cell.get_quadrature_points(_io, quadrature_type=problem.quadrature_type)
                        cell_quadrature_weights = element.cell.get_quadrature_weights(_io, quadrature_type=problem.quadrature_type)
                        x_c: ndarray = element.cell.get_centroid()
                        bdc: ndarray = element.cell.get_bounding_box()
                        element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
                        element_internal_forces = np.zeros((element.element_size,), dtype=real)
                        element_external_forces = np.zeros((element.element_size,), dtype=real)
                        for _qc in range(cell_quadrature_size):
                            _qp = element.quad_p_indices[_qc]
                            _w_q_c = cell_quadrature_weights[_qc]
                            _x_q_c = cell_quadrature_points[:, _qc]
                            # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                            transformation_gradient = element.get_transformation_gradient(total_unknown_vector, _qc)
                            material.mat_data.s1.gradients[_qp] = transformation_gradient
                            # --- INTEGRATE BEHAVIOUR LAW
                            integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
                            if integ_res != 1:
                                print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                                print("++++++++++++++++ - POINT {}".format(_x_q_c))
                                print("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                                write_out_msg("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
                                write_out_msg("++++++++++++++++ - POINT {}".format(_x_q_c))
                                write_out_msg("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
                                break_iter = True
                                break
                            else:
                                # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                w_coef = _w_q_c
                                if problem.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC, FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
                                    w_coef *= 2.0 * np.pi * _x_q_c[0]
                                element_stiffness_matrix += w_coef * (
                                        element.gradients_operators[_qc].T @ material.mat_data.K[_qp] @
                                        element.gradients_operators[_qc]
                                )
                                # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                element_internal_forces += w_coef * (
                                        element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
                                )
                                # --- COMPUTE EXTERNAL FORCES
                                v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
                                for load in problem.loads:
                                    vl = w_coef * v * load.function(time_step, _x_q_c)
                                    _re0 = load.direction * _cl
                                    _re1 = (load.direction + 1) * _cl
                                    element_external_forces[_re0:_re1] += vl
                        # GET ELEMENT CONTRIBS
                        if not break_iter:
                            # --- STAB PARAMETER CHANGE
                            stab_param = material.stabilization_parameter
                            # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                            element_stiffness_matrix += stab_param * element.stabilization_operator
                            # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                            element_internal_forces += (
                                    stab_param
                                    * element.stabilization_operator
                                    @ element.get_element_unknown_vector(total_unknown_vector)
                            )
                        return element_stiffness_matrix, element_internal_forces, element_external_forces, break_iter

                    element_stiffness_matrix, element_internal_forces, element_external_forces, break_iteration = make_cell_procedure(break_iteration)
                    local_external_forces_coefficient = 1.
                    local_tolerance = 1.e-6
                    local_iteration = 0
                    factor = material.stabilization_parameter / np.prod(bdc)
                    factor = 1.0 / np.prod(bdc)
                    # factor = 1.0
                    while local_iteration < num_local_iterations and not break_iteration:
                        R_cc = (element_internal_forces - element_external_forces)[:_c0_c]
                        # if local_external_forces_coefficient == 0.0:
                        #     local_external_forces_coefficient = 1.0
                        # local_external_forces_coefficient = 1.0 / np.prod(bdc)
                        # if local_iteration == 0:
                        #     factor = np.max(np.abs(R_cc))
                        #     factor = 1.0
                        # local_residual_evaluation = np.max(np.abs(R_cc)) * np.prod(bdc)
                        local_residual_evaluation = np.max(np.abs(R_cc)) / factor
                        # print(_element_index, local_iteration, local_residual_evaluation)
                        # local_residual_evaluation = np.max(np.abs(R_cc))
                        if local_residual_evaluation < local_tolerance:
                            break
                        else:
                            # print("local inversion")
                            element_stiffness_matrix, element_internal_forces, element_external_forces, break_iteration = make_cell_procedure(break_iteration)
                            K_cc = element_stiffness_matrix[:_c0_c, :_c0_c]
                            R_cc = (element_internal_forces - element_external_forces)[:_c0_c]
                            cell_correction = np.linalg.solve(-K_cc, R_cc)
                            total_unknown_vector[element.cell_range[0]:element.cell_range[1]] += cell_correction
                            num_cells_iterations += 1
                            local_iteration += 1
                    if not break_iteration:
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
                                        _c1 = _cl * _dx + (f_local * _dx * _fk) + (boundary_condition.direction + 1) * _fk
                                        _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
                                        _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
                                        face_lagrange = total_unknown_vector[_l0:_l1]
                                        face_displacement = total_unknown_vector[_r0:_r1]
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
                                            x_q_fp = np.copy(face_quadrature_points[:, _qf])
                                            w_q_f = face_quadrature_weights[_qf]
                                            w_coef = w_q_f
                                            if problem.field.field_type in [
                                                FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC,
                                                FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC
                                            ]:
                                                if x_q_fp[0] < 1.e-10:
                                                    x_q_fp[0] = 1.e-10
                                                w_coef *= 2.0 * np.pi * x_q_fp[0]
                                            s_f = (face_rot @ x_f)[:-1]
                                            s_q_f = (face_rot @ x_q_fp)[:-1]
                                            # coef = 2.0 * np.pi * x_q_fp[0] * w_q_f
                                            _psi_k = problem.finite_element.face_basis_k.evaluate_function(s_q_f, s_f,
                                                                                                           bdf_proj)
                                            _m_psi_psi_face += w_coef * np.tensordot(_psi_k, _psi_k, axes=0)
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
                                            x_q_fp = np.copy(face_quadrature_points[:, _qf])
                                            w_q_f = face_quadrature_weights[_qf]
                                            w_coef = w_q_f
                                            if problem.field.field_type in [
                                                FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC,
                                                FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC
                                            ]:
                                                if x_q_fp[0] < 1.e-10:
                                                    x_q_fp[0] = 1.e-10
                                                w_coef *= 2.0 * np.pi * x_q_fp[0]
                                            s_f = (face_rot @ x_f)[:-1]
                                            s_q_f = (face_rot @ x_q_fp)[:-1]
                                            # coef = 2.0 * np.pi * x_q_fp[0] * w_q_f
                                            v = problem.finite_element.face_basis_k.evaluate_function(s_q_f, s_f,
                                                                                                      bdf_proj)
                                            _v_face_imposed_displacement += (
                                                    w_coef * v * boundary_condition.function(time_step, x_q_fp)
                                            )
                                            force_item += material.lagrange_parameter * (w_coef * v @ face_lagrange[:])
                                            # _v_face_imposed_displacement += (
                                            #         _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                            # )
                                        # ----- AVANT :
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
                                            x_q_fp = np.copy(face_quadrature_points[:, _qf])
                                            w_q_f = face_quadrature_weights[_qf]
                                            w_coef = w_q_f
                                            if problem.field.field_type in [
                                                FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC,
                                                FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC
                                            ]:
                                                if x_q_fp[0] < 1.e-10:
                                                    x_q_fp[0] = 1.e-10
                                                w_coef *= 2.0 * np.pi * x_q_fp[0]
                                            s_f = (face_rot @ x_f)[:-1]
                                            s_q_f = (face_rot @ x_q_fp)[:-1]
                                            v = problem.finite_element.face_basis_k.evaluate_function(s_q_f, s_f,
                                                                                                      bdf_proj)
                                            vf = w_coef * v * boundary_condition.function(time_step, x_q_fp)
                                            # vf = _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
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
                mean_num_cells_iterations = float(num_cells_iterations) / float(problem.mesh.number_of_cells_in_mesh)
                if not break_iteration:
                    # --------------------------------------------------------------------------------------------------
                    # RESIDUAL EVALUATION
                    # --------------------------------------------------------------------------------------------------
                    if external_forces_coefficient == 0.0:
                        external_forces_coefficient = 1.0
                    residual_evaluation = np.max(np.abs(residual)) / external_forces_coefficient
                    if residual_evaluation < problem.tolerance:
                        if num_local_iterations > 0:
                            print(
                                "++++ ITER : {} | MEAN_CELLS_ITERATIONS : {:.6E} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                    str(iteration).zfill(4), mean_num_cells_iterations, residual_evaluation, problem.tolerance))
                            write_out_msg(
                                "++++ ITER : {} | MEAN_CELLS_ITERATIONS : {:.6E} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                    str(iteration).zfill(4), mean_num_cells_iterations, residual_evaluation, problem.tolerance))
                        else:
                            print(
                                "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                    str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                            write_out_msg(
                                "++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                                    str(iteration).zfill(4), residual_evaluation, problem.tolerance))
                        # --- UPDATE INTERNAL VARIABLES
                        mgis_bv.update(material.mat_data)
                        print("+ ITERATIONS : {}".format(iteration + 1))
                        write_out_msg("+ ITERATIONS : {}".format(iteration + 1))
                        for bc in problem.boundary_conditions:
                            if bc.boundary_type == BoundaryType.DISPLACEMENT:
                                bc.force_values.append(bc.force)
                                bc.time_values.append(time_step)
                        if not material.behaviour_name in ["Elasticity", "Signorini"]:
                            problem.fill_quadrature_internal_variables_output(
                                problem.res_folder_path, "INTERNAL_VARIABLES", time_step_index, time_step, material
                            )
                        problem.fill_quadrature_stress_output(problem.res_folder_path, "CAUCHY_STRESS",
                                                              time_step_index, time_step, material)
                        problem.fill_quadrature_strain_output(problem.res_folder_path, "STRAIN", time_step_index,
                                                              time_step, material)
                        problem.fill_quadrature_displacement_output(problem.res_folder_path, "QUADRATURE_DISPLACEMENT",
                                                                    time_step_index, time_step, total_unknown_vector)
                        problem.fill_node_displacement_output(problem.res_folder_path, "NODE_DISPLACEMENT",
                                                              time_step_index, time_step, total_unknown_vector)
                        for bc in problem.boundary_conditions:
                            problem.write_force_output(problem.res_folder_path, bc)
                        total_unknown_vector_previous_step = np.copy(total_unknown_vector)
                        iteration = 0
                        time_step_temp = time_step + 0.
                        time_step_index += 1
                        num_total_skeleton_time_steps += 1
                        break_iteration = True
                    elif iteration == problem.number_of_iterations - 1:
                        if num_local_iterations > 0:
                            print(
                                "++++ ITER : {} | MEAN_CELLS_ITERATIONS : {:.6E} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                                    str(iteration).zfill(4), mean_num_cells_iterations, residual_evaluation))
                            write_out_msg(
                                "++++ ITER : {} | MEAN_CELLS_ITERATIONS : {:.6E} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                                    str(iteration).zfill(4), mean_num_cells_iterations, residual_evaluation))
                        else:
                            print(
                                "++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                                    str(iteration).zfill(4), residual_evaluation))
                            write_out_msg(
                                "++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                                    str(iteration).zfill(4), residual_evaluation))
                        # print("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                        #     str(iteration).zfill(4), residual_evaluation))
                        # write_out_msg("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(
                        #     str(iteration).zfill(4), residual_evaluation))
                        total_unknown_vector = np.copy(total_unknown_vector_previous_step)
                        iteration = 0
                        problem.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                        break_iteration = True
                    else:
                        if num_local_iterations > 0:
                            print(
                                "++++ ITER : {} | MEAN_CELLS_ITERATIONS : {:.6E} | RES_MAX : {:.6E}".format(
                                    str(iteration).zfill(4), mean_num_cells_iterations, residual_evaluation))
                            write_out_msg(
                                "++++ ITER : {} | MEAN_CELLS_ITERATIONS : {:.6E} | RES_MAX : {:.6E}".format(
                                    str(iteration).zfill(4), mean_num_cells_iterations, residual_evaluation))
                        else:
                            print(
                                "++++ ITER : {} | RES_MAX : {:.6E}".format(
                                    str(iteration).zfill(4), residual_evaluation))
                            write_out_msg(
                                "++++ ITER : {} | RES_MAX : {:.6E}".format(
                                    str(iteration).zfill(4), residual_evaluation))
                        # --- SOLVE SYSTEM
                        # print("++++ ITER : {} | RES_MAX : {:.6E}".format(
                        #     str(iteration).zfill(4), residual_evaluation))
                        # write_out_msg("++++ ITER : {} | RES_MAX : {:.6E}".format(
                        #     str(iteration).zfill(4), residual_evaluation))
                        sparse_global_matrix = csr_matrix(tangent_matrix)
                        scaling_factor = np.max(np.abs(tangent_matrix))
                        correction = spsolve(sparse_global_matrix / scaling_factor, residual / scaling_factor)
                        total_unknown_vector[:_constrained_system_size] += correction
                        # --- DECONDENSATION
                        for _element_index, element in enumerate(problem.elements):
                            _nf = len(element.faces)
                            face_correction = np.zeros((_nf * _fk * _dx), dtype=real)
                            for _i_local, _i_global in enumerate(element.faces_indices):
                                _c0_fg = _i_global * (_fk * _dx)
                                _c1_fg = (_i_global + 1) * (_fk * _dx)
                                _c0_fl = _i_local * (_fk * _dx)
                                _c1_fl = (_i_local + 1) * (_fk * _dx)
                                face_correction[_c0_fl:_c1_fl] += correction[_c0_fg:_c1_fg]
                            cell_correction = np.zeros((_c0_c,), dtype=real)
                            if num_local_iterations == 0:
                                cell_correction = element.m_cell_cell_inv @ (
                                        element.v_cell - element.m_cell_faces @ face_correction
                                )
                            else:
                                _io: int = problem.finite_element.computation_integration_order
                                cell_quadrature_size = element.cell.get_quadrature_size(_io, quadrature_type=problem.quadrature_type)
                                cell_quadrature_points = element.cell.get_quadrature_points(_io, quadrature_type=problem.quadrature_type)
                                cell_quadrature_weights = element.cell.get_quadrature_weights(_io, quadrature_type=problem.quadrature_type)
                                x_c: ndarray = element.cell.get_centroid()
                                bdc: ndarray = element.cell.get_bounding_box()
                                for _qc in range(cell_quadrature_size):
                                    _qp = element.quad_p_indices[_qc]
                                    _w_q_c = cell_quadrature_weights[_qc]
                                    _x_q_c = cell_quadrature_points[:, _qc]
                                    w_coef = _w_q_c
                                    if problem.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC, FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
                                        w_coef *= 2.0 * np.pi * _x_q_c[0]
                                    element_stiffness_matrix += w_coef * (
                                            element.gradients_operators[_qc].T @ material.mat_data.K[_qp] @
                                            element.gradients_operators[_qc]
                                    )
                                    # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                                    element_internal_forces += w_coef * (
                                            element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
                                    )
                                    # --- COMPUTE EXTERNAL FORCES
                                    v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
                                    for load in problem.loads:
                                        vl = w_coef * v * load.function(time_step, _x_q_c)
                                        _re0 = load.direction * _cl
                                        _re1 = (load.direction + 1) * _cl
                                        element_external_forces[_re0:_re1] += vl
                                # GET ELEMENT CONTRIBS
                                stab_param = material.stabilization_parameter
                                # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                                element_stiffness_matrix += stab_param * element.stabilization_operator
                                # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                                element_internal_forces += (
                                        stab_param
                                        * element.stabilization_operator
                                        @ element.get_element_unknown_vector(total_unknown_vector)
                                )
                                m_cell_cell = element_stiffness_matrix[:_c0_c, :_c0_c]
                                m_cell_faces = element_stiffness_matrix[:_c0_c, _c0_c:]
                                m_faces_cell = element_stiffness_matrix[_c0_c:, :_c0_c]
                                m_faces_faces = element_stiffness_matrix[_c0_c:, _c0_c:]
                                v_cell = -element_residual[:_c0_c]
                                v_faces = -element_residual[_c0_c:]
                                m_cell_cell_inv = np.linalg.inv(m_cell_cell)
                                cell_correction = m_cell_cell_inv @ (v_cell - m_cell_faces @ face_correction)
                                # factor = 1.0
                                # face_unknowns = element.get_element_unknown_vector(total_unknown_vector)[_c0_c:]
                                # cell_unknowns = total_unknown_vector[element.cell_range[0]:element.cell_range[1]]
                                # K_update = element.stabilization_operator[:_c0_c, :_c0_c]
                                # R_update = factor * element.stabilization_operator[:_c0_c, _c0_c:] @ face_correction
                                # cell_correction = - np.linalg.inv(K_update) @ R_update
                            # --- ADDING CORRECTION TO CURRENT DISPLACEMENT
                            total_unknown_vector[element.cell_range[0]:element.cell_range[1]] += np.copy(cell_correction)
                        num_total_skeleton_iterations += 1
                        iteration += 1
                        # --------------------------------------------------------------------------------------------------
                        # ANDERSON ACCELERATE
                        # --------------------------------------------------------------------------------------------------
                        if accelerate > 0 and num_local_iterations > 0:
                            acceleration_u.accelerate(total_unknown_vector[:_constrained_system_size])
                        elif accelerate > 0 and num_local_iterations == 0:
                            acceleration_u.accelerate(total_unknown_vector)
                else:
                    print("+ SPLITTING TIME STEP")
                    write_out_msg("+ SPLITTING TIME STEP")
                    problem.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                    total_unknown_vector = np.copy(total_unknown_vector_previous_step)
                    iteration = 0
                # --- END OF ITERATION
            # --- END OF TIME STEP
        # --- END OF COMPUTATION
        write_out_msg("+ COMPUTATION NUM TIME STEPS : {:.6E}".format(num_total_skeleton_time_steps))
        write_out_msg("+ COMPUTATION NUM ITERATIONS : {:.6E}".format(num_total_skeleton_iterations))
