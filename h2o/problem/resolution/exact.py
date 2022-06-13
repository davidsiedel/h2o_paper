from h2o.problem.problem import Problem, clean_res_dir
from h2o.problem.material import Material
from h2o.h2o import *

from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def solve_newton_exact(problem: Problem, material: Material, verbose: bool = False, debug_mode: DebugMode = DebugMode.NONE):
    """

    Args:
        material:

    Returns:

    """
    clean_res_dir(problem.res_folder_path)
    _dx = problem.field.field_dimension
    _fk = problem.finite_element.face_basis_k.dimension
    _cl = problem.finite_element.cell_basis_l.dimension
    external_forces_coefficient = 1.0
    normalization_lagrange_coefficient = material.lagrange_parameter
    # ----------------------------------------------------------------------------------------------------------
    # SET SYSTEM SIZE
    # ----------------------------------------------------------------------------------------------------------
    _constrained_system_size, _system_size = problem.get_total_system_size()
    faces_unknown_vector = np.zeros((_constrained_system_size), dtype=real)
    faces_unknown_vector_previous_step = np.zeros((_constrained_system_size), dtype=real)
    residual_values = []
    correction = np.zeros((_constrained_system_size), dtype=real)
    for time_step_index, time_step in enumerate(problem.time_steps):
        # --- SET TEMPERATURE
        material.set_temperature()
        # --- PRINT DATA
        print("----------------------------------------------------------------------------------------------------")
        print("TIME_STEP : {} | LOAD_VALUE : {}".format(time_step_index, time_step))
        # --- WRITE RES FILES
        file_suffix = "{}".format(time_step_index).zfill(6)
        problem.create_vertex_res_files(problem.res_folder_path, file_suffix)
        problem.create_quadrature_points_res_files(problem.res_folder_path, file_suffix, material)
        # correction = np.zeros((_constrained_system_size),dtype=real)
        # --- RESET DISPLACEMENT
        reset_displacement = False
        if reset_displacement:
            faces_unknown_vector = np.zeros((_constrained_system_size), dtype=real)
            for element in problem.elements:
                element.cell_unknown_vector = np.zeros((_dx * _cl,), dtype=real)
        for iteration in range(problem.number_of_iterations):
            # --------------------------------------------------------------------------------------------------
            # SET SYSTEM MATRIX AND VECTOR
            # --------------------------------------------------------------------------------------------------
            tangent_matrix = np.zeros((_constrained_system_size, _constrained_system_size), dtype=real)
            residual = np.zeros((_constrained_system_size), dtype=real)
            # --------------------------------------------------------------------------------------------------
            # SET TIME INCREMENT
            # --------------------------------------------------------------------------------------------------
            if time_step_index == 0:
                _dt = time_step
            else:
                _dt = time_step - problem.time_steps[time_step_index - 1]
                _dt = np.float64(_dt)
            # --------------------------------------------------------------------------------------------------
            # FOR ELEMENT LOOP
            # --------------------------------------------------------------------------------------------------
            _qp = 0
            stab_coef = 1.0
            stab_inf = np.inf
            iter_face_constraint = 0
            if False:
                for element in problem.elements:
                    for _qc in range(len(element.cell.quadrature_weights)):
                        _w_q_c = element.cell.quadrature_weights[_qc]
                        _x_q_c = element.cell.quadrature_points[:, _qc]
                        # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                        transformation_gradient = element.get_transformation_gradient(
                            problem.field, problem.finite_element, faces_unknown_vector, _qc
                        )
                        material.mat_data.s1.gradients[_qp] = transformation_gradient
                        # --- INTEGRATE BEHAVIOUR LAW
                        integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
                        eigen_values = np.linalg.eig(material.mat_data.K[_qp])[0]
                        stab_elem = np.min(eigen_values) / material.stabilization_parameter
                        if stab_elem < stab_inf and stab_elem > 1.0:
                            stab_inf = stab_elem
                            stab_coef = stab_elem
                        _qp += 1
                    mgis_bv.revert(material.mat_data)
            _qp = 0
            for _element_index, element in enumerate(problem.elements):
                cell_quadrature_size = element.cell.get_quadrature_size(
                    problem.finite_element.construction_integration_order, quadrature_type=problem.quadrature_type
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    problem.finite_element.construction_integration_order, quadrature_type=problem.quadrature_type
                )
                cell_quadrature_weights = element.cell.get_quadrature_weights(
                    problem.finite_element.construction_integration_order, quadrature_type=problem.quadrature_type
                )
                x_c = element.cell.get_centroid()
                h_c = element.cell.get_diameter()
                print("ELEMENT : {}".format(_element_index))
                # --- GET FACES LOCAL CORRECTION
                local_faces_correction = np.zeros((len(element.faces) * _dx * _fk), dtype=real)
                for _f_local, _f_global in enumerate(element.faces_indices):
                    local_faces_correction[_f_local * _dx * _fk:(_f_local + 1) * _dx * _fk] += (
                        correction[_f_global * _dx * _fk:(_f_global + 1) * _dx * _fk]
                    )
                # --- LOCAL NEWTON
                number_of_local_iterations = 20
                # local_tolerance = problem.tolerance
                local_tolerance = 1.e-4
                R_cell_value_previous = np.inf
                for _local_iteration in range(number_of_local_iterations):
                    _nf = len(element.faces)
                    _c0_c = _dx * _cl
                    # --- INITIALIZE MATRIX AND VECTORS
                    element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
                    element_internal_forces = np.zeros((element.element_size,), dtype=real)
                    element_external_forces = np.zeros((element.element_size,), dtype=real)
                    # --- RUN OVER EACH QUADRATURE POINT
                    _local_qp = 0
                    for _qc in range(cell_quadrature_size):
                        _w_q_c = cell_quadrature_weights[_qc]
                        _x_q_c = cell_quadrature_points[:, _qc]
                        # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                        transformation_gradient = element.get_transformation_gradient(
                            faces_unknown_vector, _qc
                        )
                        material.mat_data.s1.gradients[_qp] = transformation_gradient
                        # --- INTEGRATE BEHAVIOUR LAW
                        integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
                        # stored_energies', 'dissipated_energies', 'internal_state_variables
                        # print(material.mat_data.s1.internal_state_variables)
                        # --- VOLUMETRIC FORCES
                        v = problem.finite_element.cell_basis_l.evaluate_function(
                            _x_q_c, x_c, h_c
                        )
                        for load in problem.loads:
                            vl = _w_q_c * v * load.function(time_step, _x_q_c)
                            _re0 = load.direction * _cl
                            _re1 = (load.direction + 1) * _cl
                            element_external_forces[_re0:_re1] += vl
                        # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                        element_stiffness_matrix += _w_q_c * (
                                element.gradients_operators[_qc].T
                                @ material.mat_data.K[_qp]
                                @ element.gradients_operators[_qc]
                        )
                        # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                        element_internal_forces += _w_q_c * (
                                element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
                        )
                        _qp += 1
                        _local_qp += 1
                    if verbose:
                        print("ELEM : {} | INTERNAL_FORCES_BEFORE STAB : \n {}".format(_element_index,
                                                                                       element_internal_forces))
                    # --- STAB PARAMETER CHANGE
                    stab_param = stab_coef * material.stabilization_parameter
                    # if check and iteration > 0:
                    #     numerical_element_stiffness_matrix = (
                    #                                                  numerical_forces_bs_0[_element_index] -
                    #                                                  numerical_forces_bs_1[_element_index]
                    #                                          ) / (2.0 * noise)
                    #     check_num = np.max(
                    #         np.abs(numerical_element_stiffness_matrix - element_stiffness_matrix)
                    #     ) / np.max(np.abs(element_stiffness_matrix))
                    #     if check_num > noise_tolerance:
                    #         print(
                    #             "AT ELEM : {} | CHECK_BEFORE_STAB : {}".format(str(_element_index).zfill(6), check_num)
                    #         )
                    # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                    element_stiffness_matrix += stab_param * element.stabilization_operator
                    # element_stiffness_matrix -= stab_param * element.stabilization_operator
                    # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                    element_internal_forces += (
                            stab_param
                            * element.stabilization_operator
                            @ element.get_element_unknown_vector(faces_unknown_vector)
                    )
                    # --- MATRIX VECTOR DECOMPOSITION
                    K_cc = element_stiffness_matrix[:_c0_c, :_c0_c]
                    K_cf = element_stiffness_matrix[:_c0_c, _c0_c:]
                    # local_external_forces_coefficient = np.max(np.abs(element_external_forces))
                    element_residual = element_internal_forces - element_external_forces
                    # if local_external_forces_coefficient == 0.:
                    #     local_external_forces_coefficient = 1.0
                    # local_external_forces_coefficient = normalization_lagrange_coefficient
                    # local_external_forces_coefficient = 1./element.cell.shape.volume
                    R_cc = element_residual[:_c0_c]
                    if iteration == 0:
                        local_external_forces_coefficient = 1.0
                        cell_correction = np.ones((_cl * _dx,), dtype=real)
                    if _local_iteration == 0 and iteration > 0:
                        cell_correction = np.ones((_cl * _dx,), dtype=real)
                        local_external_forces_coefficient = np.max(np.abs(R_cc))
                        # local_external_forces_coefficient = 1.
                        # if local_external_forces_coefficient == 0.0 or local_external_forces_coefficient < local_tolerance:
                        if local_external_forces_coefficient == 0.0:
                            local_external_forces_coefficient = 1.0
                    # R_cell = R_cc - K_cf @ local_faces_correction
                    # print("LOCAL_EXTERNAL_FORCES_COEF : {}".format(local_external_forces_coefficient))
                    R_cell = R_cc
                    local_residual_evaluation = np.max(np.abs(R_cell) / local_external_forces_coefficient)
                    # local_residual_evaluation = np.max(np.abs(cell_correction))
                    # if local_residual_evaluation > R_cell_value_previous:
                    #     print("!!!! RESIUDAL INCREASE :\n {}".format(element.cell.quadrature_points))
                    # R_cell_value_previous = local_residual_evaluation
                    print("LOCAL ITER : {} | RES MAX : {}".format(_local_iteration, local_residual_evaluation))
                    if local_residual_evaluation < local_tolerance:
                        break
                    else:
                        cell_correction = np.linalg.solve(-K_cc, R_cell)
                        element.cell_unknown_vector += cell_correction
                        if _local_iteration != number_of_local_iterations - 1:
                            _qp -= _local_qp
                # element_internal_forces -= (
                #     stab_param
                #     * element.stabilization_operator
                #     @ element.get_element_unknown_vector(problem.field, problem.finite_element, faces_unknown_vector)
                # )
                # --------------------------------------------------------------------------------------------------
                # CONDENSATION
                # --------------------------------------------------------------------------------------------------
                m_cell_cell = element_stiffness_matrix[:_c0_c, :_c0_c]
                m_cell_faces = element_stiffness_matrix[:_c0_c, _c0_c:]
                m_faces_cell = element_stiffness_matrix[_c0_c:, :_c0_c]
                m_faces_faces = element_stiffness_matrix[_c0_c:, _c0_c:]
                # v_cell = element_residual[:_c0_c]
                # v_faces = element_residual[_c0_c:]
                v_cell = -element_residual[:_c0_c]
                v_faces = -element_residual[_c0_c:]
                m_cell_cell_inv = np.linalg.inv(m_cell_cell)
                if debug_mode == 0:
                    print("TANGENT MATRIX IN CONDENSATION COND :")
                    print("{}".format(np.linalg.cond(m_cell_cell)))
                # ge = m_faces_cell @ m_cell_cell_inv
                # gd = (m_faces_cell @ m_cell_cell_inv) @ m_cell_faces
                K_cond = m_faces_faces - ((m_faces_cell @ m_cell_cell_inv) @ m_cell_faces)
                R_cond = v_faces - (m_faces_cell @ m_cell_cell_inv) @ v_cell
                # --- SET CONDENSATION/DECONDENSATION MATRICES
                # element.m_cell_cell_inv = m_cell_cell_inv
                # element.m_cell_faces = m_cell_faces
                # element.v_cell = v_cell
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
                # if check and iteration > 0:
                #     numerical_element_stiffness_matrix = (
                #                                                  numerical_forces_as_0[_element_index] -
                #                                                  numerical_forces_as_1[_element_index]
                #                                          ) / (2.0 * noise)
                #     check_num = np.max(
                #         np.abs(numerical_element_stiffness_matrix - element_stiffness_matrix)
                #     ) / np.max(np.abs(element_stiffness_matrix))
                #     if check_num > noise_tolerance:
                #         print(
                #             "AT ELEM : {} | CHECK_BEFORE_STAB : {}".format(str(_element_index).zfill(6), check_num)
                #         )
                if verbose:
                    pass
                    # print("ELEM : {} | INTERNAL_FORCES_AFTER STAB : \n {}".format(_element_index,
                    #                                                               element_internal_forces))
                    # _iv0 = _element_index * len(element.cell.quadrature_weights)
                    # _iv1 = (_element_index + 1) * len(element.cell.quadrature_weights)
                    # print("ELEM : {} | DISPLACEMENT S0 : \n {}".format(_element_index,
                    #                                                    element.get_element_unknown_vector(problem.field,
                    #                                                                                       problem.finite_element,
                    #                                                                                       faces_unknown_vector_previous_step)))
                    # print("ELEM : {} | GRADIENTS S0 : \n {}".format(_element_index,
                    #                                                 material.mat_data.s0.gradients[_iv0:_iv1]))
                    # print("ELEM : {} | DISPLACEMENT S1 : \n {}".format(_element_index,
                    #                                                    element.get_element_unknown_vector(problem.field,
                    #                                                                                       problem.finite_element,
                    #                                                                                       faces_unknown_vector)))
                    # print("ELEM : {} | GRADIENTS S1 : \n {}".format(_element_index,
                    #                                                 material.mat_data.s1.gradients[_iv0:_iv1]))
                    # if not material.behaviour_name == "Elasticity":
                    #     print("ELEM : {} | INTERNAL_STATE_VARIABLES S0 : \n {}".format(_element_index,
                    #                                                                    material.mat_data.s0.internal_state_variables[
                    #                                                                    _iv0:_iv1]))
                    #     print("ELEM : {} | INTERNAL_STATE_VARIABLES S1 : \n {}".format(_element_index,
                    #                                                                    material.mat_data.s1.internal_state_variables[
                    #                                                                    _iv0:_iv1]))
                # --- BOUNDARY CONDITIONS
                for boundary_condition in problem.boundary_conditions:
                    # --- DISPLACEMENT CONDITIONS
                    if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                        for f_local, f_global in enumerate(element.faces_indices):
                            if (
                                    f_global
                                    in problem.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]
                            ):
                                _l0 = _system_size + iter_face_constraint * _fk
                                _l1 = _system_size + (iter_face_constraint + 1) * _fk
                                _c0 = _cl * _dx + (f_local * _dx * _fk) + boundary_condition.direction * _fk
                                _c1 = _cl * _dx + (f_local * _dx * _fk) + (boundary_condition.direction + 1) * _fk
                                _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
                                _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
                                # -------
                                # face_displacement = element_unknown_increment[_r0:_r1]
                                # face_displacement = np.copy(element_unknown_increment[_c0:_c1])
                                # face_displacement = element_unknown_increment[_c0:_c1]
                                face_lagrange = faces_unknown_vector[_l0:_l1]
                                face_displacement = faces_unknown_vector[_r0:_r1]
                                _m_psi_psi_face = np.zeros((_fk, _fk), dtype=real)
                                _v_face_imposed_displacement = np.zeros((_fk,), dtype=real)
                                face = element.faces[f_local]
                                x_f = face.get_centroid()
                                h_f = face.get_diameter()
                                face_rot = face.get_rotation_matrix()
                                face_quadrature_size = face.get_quadrature_size(
                                    problem.finite_element.construction_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_points = face.get_quadrature_points(
                                    problem.finite_element.construction_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_weights = face.get_quadrature_weights(
                                    problem.finite_element.construction_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                for _qf in range(face_quadrature_size):
                                    _x_q_f = face_quadrature_points[:, _qf]
                                    _w_q_f = face_quadrature_weights[_qf]
                                    _s_q_f = (face_rot @ _x_q_f)[:-1]
                                    _s_f = (face_rot @ x_f)[:-1]
                                    v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, h_f)
                                    _v_face_imposed_displacement += (
                                            _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                    )
                                    _psi_k = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, h_f)
                                    _m_psi_psi_face += _w_q_f * np.tensordot(_psi_k, _psi_k, axes=0)
                                _m_psi_psi_face_inv = np.linalg.inv(_m_psi_psi_face)
                                if debug_mode == 0:
                                    print("FACE MASS MATRIX IN DIRICHLET BOUND COND :")
                                    print("{}".format(np.linalg.cond(_m_psi_psi_face)))
                                imposed_face_displacement = _m_psi_psi_face_inv @ _v_face_imposed_displacement
                                face_displacement_difference = face_displacement - imposed_face_displacement
                                # --- LAGRANGE INTERNAL FORCES PART
                                # element_internal_forces[_c0:_c1] += (
                                #     normalization_lagrange_coefficient * face_lagrange
                                # )
                                # residual[_r0:_r1] += (
                                #     normalization_lagrange_coefficient * face_lagrange
                                # )
                                residual[_r0:_r1] -= (
                                        normalization_lagrange_coefficient * face_lagrange
                                )
                                # --- LAGRANGE MULTIPLIERS PART
                                # residual[_l0:_l1] += (
                                #     normalization_lagrange_coefficient * face_displacement_difference
                                # )
                                residual[_l0:_l1] -= (
                                        normalization_lagrange_coefficient * face_displacement_difference
                                )
                                # --- LAGRANGE MATRIX PART
                                tangent_matrix[_l0:_l1, _r0:_r1] += normalization_lagrange_coefficient * np.eye(_fk,
                                                                                                                dtype=real)
                                tangent_matrix[_r0:_r1, _l0:_l1] += normalization_lagrange_coefficient * np.eye(_fk,
                                                                                                                dtype=real)
                                # --- SET EXTERNAL FORCES COEFFICIENT
                                lagrange_external_forces = (
                                        normalization_lagrange_coefficient * imposed_face_displacement
                                )
                                if np.max(np.abs(lagrange_external_forces)) > external_forces_coefficient:
                                    external_forces_coefficient = np.max(np.abs(lagrange_external_forces))
                                iter_face_constraint += 1
                    elif boundary_condition.boundary_type == BoundaryType.PRESSURE:
                        for f_local, f_global in enumerate(element.faces_indices):
                            if (
                                    f_global
                                    in problem.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]
                            ):
                                face = element.faces[f_local]
                                x_f = face.get_centroid()
                                h_f = face.get_diameter()
                                face_rot = face.get_rotation_matrix()
                                face_quadrature_size = face.get_quadrature_size(
                                    problem.finite_element.construction_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_points = face.get_quadrature_points(
                                    problem.finite_element.construction_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_weights = face.get_quadrature_weights(
                                    problem.finite_element.construction_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                for _qf in range(face_quadrature_size):
                                    # _h_f = element.faces[f_local].shape.diameter
                                    # _x_f = element.faces[f_local].shape.centroid
                                    _x_q_f = face_quadrature_points[:, _qf]
                                    _w_q_f = face_quadrature_weights[_qf]
                                    _s_q_f = (face_rot @ _x_q_f)[:-1]
                                    _s_f = (face_rot @ x_f)[:-1]
                                    v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, h_f)
                                    vf = _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                    # _c0 = _dx * _cl + f_local * _dx * _fk + boundary_condition.direction * _fk
                                    # _c1 = _dx * _cl + f_local * _dx * _fk + (boundary_condition.direction + 1) * _fk
                                    _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
                                    _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
                                    # element_external_forces[_c0:_c1] += vf
                                    # residual[_r0:_r1] -= vf
                                    residual[_r0:_r1] += vf
                                    if np.max(np.abs(vf)) > external_forces_coefficient:
                                        external_forces_coefficient = np.max(np.abs(vf))
            # --------------------------------------------------------------------------------------------------
            # RESIDUAL EVALUATION
            # --------------------------------------------------------------------------------------------------
            if external_forces_coefficient == 0.0:
                external_forces_coefficient = 1.0
            residual_evaluation = np.max(np.abs(residual)) / external_forces_coefficient
            print("ITER : {} | RES_MAX : {}".format(str(iteration).zfill(4), residual_evaluation))
            # print("ITER : {} | =====================================================================".format(str(iteration).zfill(4)))
            # residual_values.append(residual_evaluation)
            # if residual_evaluation < problem.tolerance:
            if residual_evaluation < problem.tolerance or iteration == problem.number_of_iterations - 1:
                # ----------------------------------------------------------------------------------------------
                # UPDATE INTERNAL VARIABLES
                # ----------------------------------------------------------------------------------------------
                mgis_bv.update(material.mat_data)
                print("ITERATIONS : {}".format(iteration + 1))
                problem.write_vertex_res_files(problem.res_folder_path, file_suffix, faces_unknown_vector)
                problem.write_quadrature_points_res_files(problem.res_folder_path, file_suffix, material, faces_unknown_vector)
                faces_unknown_vector_previous_step += faces_unknown_vector
                residual_values.append(residual_evaluation)
                break
            else:
                # ----------------------------------------------------------------------------------------------
                # SOLVE SYSTEM
                # ----------------------------------------------------------------------------------------------
                if debug_mode == 0:
                    print("K GLOBAL COND")
                    print("{}".format(np.linalg.cond(tangent_matrix)))
                # sparse_global_matrix = csr_matrix(-tangent_matrix)
                sparse_global_matrix = csr_matrix(tangent_matrix)
                correction = spsolve(sparse_global_matrix, residual)
                print("CORRECTION :")
                # print(correction)
                # correction = np.linalg.solve(-tangent_matrix, residual)
                faces_unknown_vector += correction
                if verbose:
                    print("R_K_RES : \n {}".format(residual - tangent_matrix @ correction))
                    print("R_K_RES_NORMALIZED : \n {}".format(
                        (residual - tangent_matrix @ correction) / external_forces_coefficient))
    plt.plot(range(len(residual_values)), residual_values, label="residual")
    plt.plot(range(len(residual_values)), [problem.tolerance for local_i in range(len(residual_values))],
             label="tolerance")
    plt.ylabel("resiudal after convergence")
    plt.xlabel("time step index")
    plt.title("evolution of the residual")
    plt.legend()
    plt.gcf().subplots_adjust(left=0.15)
    plt.show()
    return