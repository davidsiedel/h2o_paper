import numpy as np

from h2o.problem.problem import Problem, clean_res_dir
from h2o.problem.material import Material
from h2o.h2o import *

from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def solve_newton_2(problem: Problem, material: Material, verbose: bool = False, debug_mode: DebugMode = DebugMode.NONE):
    time_step_index_count = 0
    clean_res_dir(problem.res_folder_path)
    problem.create_output(problem.res_folder_path)
    _dx = problem.field.field_dimension
    _fk = problem.finite_element.face_basis_k.dimension
    _cl = problem.finite_element.cell_basis_l.dimension
    external_forces_coefficient = 1.0
    normalization_lagrange_coefficient = material.lagrange_parameter
    lagrange_parameter_buffer = material.lagrange_parameter
    # ----------------------------------------------------------------------------------------------------------
    # SET SYSTEM SIZE
    # ----------------------------------------------------------------------------------------------------------
    _constrained_system_size, _system_size = problem.get_total_system_size()
    faces_unknown_vector = np.zeros((_constrained_system_size), dtype=real)
    faces_unknown_vector_previous_step = np.zeros((_constrained_system_size), dtype=real)
    residual_values = []
    time_step_temp = problem.time_steps[0]
    initial_time_steps = [ff for ff in problem.time_steps]
    _STAB_PARAM_FINAL = 1.0
    _STAB_PARAM_FACTOR = 1.0
    time_step_index = 0
    while time_step_index < len(problem.time_steps):
        time_step = problem.time_steps[time_step_index]
    # for time_step_index, time_step in enumerate(problem.time_steps):
    #     local_time_steps = [time_step]
        # for local_time_step_index, local_time_step in enumerate(local_time_steps):
        #     print("0K")
        # --- SET TEMPERATURE
        material.set_temperature()
        # --- PRINT DATA
        print("----------------------------------------------------------------------------------------------------")
        print("+ TIME_STEP : {} | LOAD_VALUE : {}".format(time_step_index, time_step))
        # --- WRITE RES FILES
        # if time_step in initial_time_steps:
        # file_suffix = "{}".format(time_step_index).zfill(6)
        # problem.create_vertex_res_files(problem.res_folder_path, file_suffix)
        # problem.create_quadrature_points_res_files(problem.res_folder_path, file_suffix, material)
        for iteration in range(problem.number_of_iterations):
            inte_res_failures = []
            normalization_lagrange_coefficient = lagrange_parameter_buffer
            _MIN_EIGVALS = np.inf
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
            # _dt = 0.0
            # _dt = np.float64(0.0)
            # --------------------------------------------------------------------------------------------------
            # FOR ELEMENT LOOP
            # --------------------------------------------------------------------------------------------------
            _qp = 0
            stab_coef = 1.0
            iter_face_constraint = 0
            boundary_force: float = 0.0
            boundary_num_faces: int = 0
            for boundary_condition in problem.boundary_conditions:
                if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                    boundary_condition.force = 0.0
            for _element_index, element in enumerate(problem.elements):
                # cell_quadrature_size = element.cell.get_quadrature_size(
                #     problem.finite_element.construction_integration_order, quadrature_type=problem.quadrature_type
                # )
                # cell_quadrature_points = element.cell.get_quadrature_points(
                #     problem.finite_element.construction_integration_order, quadrature_type=problem.quadrature_type
                # )
                # cell_quadrature_weights = element.cell.get_quadrature_weights(
                #     problem.finite_element.construction_integration_order, quadrature_type=problem.quadrature_type
                # )
                cell_quadrature_size = element.cell.get_quadrature_size(
                    problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                )
                cell_quadrature_weights = element.cell.get_quadrature_weights(
                    problem.finite_element.computation_integration_order, quadrature_type=problem.quadrature_type
                )
                x_c = element.cell.get_centroid()
                h_c = element.cell.get_diameter()
                bdc = element.cell.get_bounding_box()
                _nf = len(element.faces)
                _c0_c = _dx * _cl
                # --- INITIALIZE MATRIX AND VECTORS
                element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
                element_internal_forces = np.zeros((element.element_size,), dtype=real)
                element_external_forces = np.zeros((element.element_size,), dtype=real)
                # --- RUN OVER EACH QUADRATURE POINT
                for _qc in range(cell_quadrature_size):
                    _w_q_c = cell_quadrature_weights[_qc]
                    # _w_q_c = np.abs(cell_quadrature_weights[_qc])
                    _x_q_c = cell_quadrature_points[:, _qc]
                    # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
                    transformation_gradient = element.get_transformation_gradient(faces_unknown_vector, _qc)
                    material.mat_data.s1.gradients[_qp] = transformation_gradient
                    # --- INTEGRATE BEHAVIOUR LAW
                    integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
                    if integ_res != 1:
                        inte_res_failures.append([_qp, material.mat_data.s1.gradients])
                        # print("INTEG RES : {} @ QUAD POINT {}".format(integ_res, _qp))
                    # stored_energies', 'dissipated_energies', 'internal_state_variables
                    # print(material.mat_data.s1.internal_state_variables)
                    # --- VOLUMETRIC FORCES
                    # v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, h_c)
                    v = problem.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
                    for load in problem.loads:
                        vl = _w_q_c * v * load.function(time_step, _x_q_c)
                        _re0 = load.direction * _cl
                        _re1 = (load.direction + 1) * _cl
                        element_external_forces[_re0:_re1] += vl
                    # --- UPDATE STAB PARAM
                    if False:
                        eig_vals, _ = np.linalg.eig(material.mat_data.K[_qp])
                        local_factor = np.min(eig_vals)/material.stabilization_parameter
                        if debug_mode == DebugMode.LIGHT:
                            print("K : \n{}".format(material.mat_data.K[_qp]))
                            print("min eig : \n{:.6E}".format(np.min(eig_vals)))
                            print("max eig : \n{:.6E}".format(np.max(eig_vals)))
                            print("2 mu : \n{:.6E}".format(material.stabilization_parameter))
                            print("local_factor : {}".format(local_factor))
                        if local_factor > _STAB_PARAM_FACTOR:
                            _STAB_PARAM_FACTOR = local_factor
                        if np.min(eig_vals) < _MIN_EIGVALS:
                            _MIN_EIGVALS = np.min(eig_vals)
                    # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                    mat_K = material.mat_data.K[_qp]
                    if iteration == 0:
                        mat_K = np.array(
                            [[2.71131414e+11, 1.10743817e+11, 1.10743817e+11, 0.00000000e+00, 0.00000000e+00],
                             [1.10743817e+11, 2.71131414e+11, 1.10743817e+11, 0.00000000e+00, 0.00000000e+00],
                             [1.10743817e+11, 1.10743817e+11, 2.71131414e+11, 0.00000000e+00, 0.00000000e+00],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.01937984e+10, 8.01937984e+10],
                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.01937984e+10, 8.01937984e+10],]
                        )
                    element_stiffness_matrix += _w_q_c * (
                        element.gradients_operators[_qc].T @ material.mat_data.K[_qp] @ element.gradients_operators[_qc]
                    )
                    # element_stiffness_matrix += _w_q_c * (
                    #     element.gradients_operators[_qc].T @ mat_K @ element.gradients_operators[_qc]
                    # )
                    # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                    element_internal_forces += _w_q_c * (
                        element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
                    )
                    _qp += 1
                if debug_mode == DebugMode.LIGHT:
                    print(
                        "ELEM : {} | INTERNAL_FORCES_BEFORE STAB : \n {}".format(
                            _element_index, element_internal_forces
                        )
                    )
                if verbose:
                    print(
                        "ELEM : {} | INTERNAL_FORCES_BEFORE STAB : \n {}".format(
                            _element_index, element_internal_forces
                        )
                    )
                # --- STAB PARAMETER CHANGE
                stab_param = stab_coef * material.stabilization_parameter
                # stab_param = _STAB_PARAM_FINAL * material.stabilization_parameter
                # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                element_stiffness_matrix += stab_param * element.stabilization_operator
                # element_stiffness_matrix -= stab_param * element.stabilization_operator
                # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
                element_internal_forces += (
                    stab_param
                    * element.stabilization_operator
                    @ element.get_element_unknown_vector(faces_unknown_vector)
                )
                # element_internal_forces -= (
                #     stab_param
                #     * element.stabilization_operator
                #     @ element.get_element_unknown_vector(problem.field, problem.finite_element, faces_unknown_vector)
                # )
                # if verbose:
                #     print(
                #         "ELEM : {} | INTERNAL_FORCES_AFTER STAB : \n {}".format(_element_index, element_internal_forces)
                #     )
                #     _iv0 = _element_index * len(element.cell.quadrature_weights)
                #     _iv1 = (_element_index + 1) * len(element.cell.quadrature_weights)
                #     print(
                #         "ELEM : {} | DISPLACEMENT S0 : \n {}".format(
                #             _element_index,
                #             element.get_element_unknown_vector(
                #                 problem.field, problem.finite_element, faces_unknown_vector_previous_step
                #             ),
                #         )
                #     )
                #     print(
                #         "ELEM : {} | GRADIENTS S0 : \n {}".format(
                #             _element_index, material.mat_data.s0.gradients[_iv0:_iv1]
                #         )
                #     )
                #     print(
                #         "ELEM : {} | DISPLACEMENT S1 : \n {}".format(
                #             _element_index,
                #             element.get_element_unknown_vector(
                #                 problem.field, problem.finite_element, faces_unknown_vector
                #             ),
                #         )
                #     )
                #     print(
                #         "ELEM : {} | GRADIENTS S1 : \n {}".format(
                #             _element_index, material.mat_data.s1.gradients[_iv0:_iv1]
                #         )
                #     )
                #     if not material.behaviour_name == "Elasticity":
                #         print(
                #             "ELEM : {} | INTERNAL_STATE_VARIABLES S0 : \n {}".format(
                #                 _element_index, material.mat_data.s0.internal_state_variables[_iv0:_iv1]
                #             )
                #         )
                #         print(
                #             "ELEM : {} | INTERNAL_STATE_VARIABLES S1 : \n {}".format(
                #                 _element_index, material.mat_data.s1.internal_state_variables[_iv0:_iv1]
                #             )
                #         )
                # --- BOUNDARY CONDITIONS
                for boundary_condition in problem.boundary_conditions:
                    # --- DISPLACEMENT CONDITIONS
                    if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                        for f_local, f_global in enumerate(element.faces_indices):
                            if f_global in problem.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]:
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
                                # _v_face_associated_force = np.zeros((_fk,), dtype=real)
                                face = element.faces[f_local]
                                x_f = face.get_centroid()
                                h_f = face.get_diameter()
                                bdf = face.get_bounding_box()
                                face_rot = face.get_rotation_matrix()
                                # face_quadrature_size = face.get_quadrature_size(
                                #     problem.finite_element.construction_integration_order,
                                #     quadrature_type=problem.quadrature_type,
                                # )
                                # face_quadrature_points = face.get_quadrature_points(
                                #     problem.finite_element.construction_integration_order,
                                #     quadrature_type=problem.quadrature_type,
                                # )
                                # face_quadrature_weights = face.get_quadrature_weights(
                                #     problem.finite_element.construction_integration_order,
                                #     quadrature_type=problem.quadrature_type,
                                # )
                                face_quadrature_size = face.get_quadrature_size(
                                    problem.finite_element.computation_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_points = face.get_quadrature_points(
                                    problem.finite_element.computation_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_weights = face.get_quadrature_weights(
                                    problem.finite_element.computation_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                force_item = 0.0
                                for _qf in range(face_quadrature_size):
                                    # _h_f = element.faces[f_local].shape.diameter
                                    # _x_f = element.faces[f_local].shape.centroid
                                    _x_q_f = face_quadrature_points[:, _qf]
                                    _w_q_f = face_quadrature_weights[_qf]
                                    _s_q_f = (face_rot @ _x_q_f)[:-1]
                                    _s_f = (face_rot @ x_f)[:-1]
                                    bdf_proj = (face_rot @ bdf)[:-1]
                                    bdf_proj = face.get_face_bounding_box()
                                    # v = problem.finite_element.face_basis_k.evaluate_function(
                                    #     _x_q_f,
                                    #     element.faces[f_local].shape.centroid,
                                    #     element.faces[f_local].shape.diameter,
                                    # )
                                    # v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, h_f)
                                    v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, bdf_proj)
                                    _v_face_imposed_displacement += (
                                        _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                    )
                                    force_item += (
                                            # _w_q_f * v @ face_lagrange / bdf_proj[0]
                                            _w_q_f * v @ face_lagrange
                                            # v @ face_lagrange
                                            # face_lagrange[0]
                                        )
                                    # if boundary_condition.boundary_name == "RIGHT":
                                    #     boundary_force += (
                                    #         _w_q_f * v @ face_lagrange / bdf_proj[0]
                                    #         # v @ face_lagrange
                                    #         # face_lagrange[0]
                                    #     )
                                    #     print("FACE LAG")
                                    #     print(face_lagrange)
                                    #     print("FACE INTEGR")
                                    #     print(boundary_force)
                                    # boundary_force += face_lagrange[0]/face.get_face_bounding_box()[0]
                                    # _m_psi_psi_face += blocks.get_face_mass_matrix_in_face(
                                    #     element.faces[f_local],
                                    #     problem.finite_element.face_basis_k,
                                    #     problem.finite_element.face_basis_k,
                                    #     _x_q_f,
                                    #     _w_q_f,
                                    # )
                                    # _psi_k = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, h_f)
                                    _psi_k = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, bdf_proj)
                                    _m_psi_psi_face += _w_q_f * np.tensordot(_psi_k, _psi_k, axes=0)
                                force_item *= normalization_lagrange_coefficient/face.get_diameter()
                                # force_item *= normalization_lagrange_coefficient * face.get_diameter()
                                # force_item *= normalization_lagrange_coefficient
                                # force_item *= normalization_lagrange_coefficient/face.get_bounding_box()[boundary_condition.direction]
                                # force_item *= normalization_lagrange_coefficient
                                boundary_condition.force += force_item
                                _m_psi_psi_face_inv = np.linalg.inv(_m_psi_psi_face)
                                if debug_mode == 0:
                                    print("FACE MASS MATRIX IN DIRICHLET BOUND COND :")
                                    print("{}".format(np.linalg.cond(_m_psi_psi_face)))
                                imposed_face_displacement = _m_psi_psi_face_inv @ _v_face_imposed_displacement
                                face_displacement_difference = face_displacement - imposed_face_displacement
                                # print(face_lagrange)
                                # --- LAGRANGE INTERNAL FORCES PART
                                element_internal_forces[_c0:_c1] += normalization_lagrange_coefficient * face_lagrange
                                # --- LAGRANGE MULTIPLIERS PART
                                # residual[_l0:_l1] += (
                                #     normalization_lagrange_coefficient * face_displacement_difference
                                # )
                                residual[_l0:_l1] -= normalization_lagrange_coefficient * face_displacement_difference
                                # --- LAGRANGE MATRIX PART
                                tangent_matrix[_l0:_l1, _r0:_r1] += normalization_lagrange_coefficient * np.eye(
                                    _fk, dtype=real
                                )
                                tangent_matrix[_r0:_r1, _l0:_l1] += normalization_lagrange_coefficient * np.eye(
                                    _fk, dtype=real
                                )
                                # --- SET EXTERNAL FORCES COEFFICIENT
                                lagrange_external_forces = (
                                    normalization_lagrange_coefficient * imposed_face_displacement
                                )
                                if np.max(np.abs(lagrange_external_forces)) > external_forces_coefficient:
                                    external_forces_coefficient = np.max(np.abs(lagrange_external_forces))
                                iter_face_constraint += 1
                                # bc_face_count += 1
                        # if bc_face_count > 0:
                            # print(boundary_condition.boundary_name)
                            # print(boundary_force)
                            # print(bc_face_count)
                            # boundary_condition.force += boundary_force/bc_face_count
                            # boundary_condition.force += boundary_force
                        # forces_list[bc_count].append(boundary_force/bc_face_count)
                    elif boundary_condition.boundary_type == BoundaryType.PRESSURE:
                        for f_local, f_global in enumerate(element.faces_indices):
                            if f_global in problem.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]:
                                face = element.faces[f_local]
                                x_f = face.get_centroid()
                                h_f = face.get_diameter()
                                bdf = face.get_bounding_box()
                                face_rot = face.get_rotation_matrix()
                                # face_quadrature_size = face.get_quadrature_size(
                                #     problem.finite_element.construction_integration_order,
                                #     quadrature_type=problem.quadrature_type,
                                # )
                                # face_quadrature_points = face.get_quadrature_points(
                                #     problem.finite_element.construction_integration_order,
                                #     quadrature_type=problem.quadrature_type,
                                # )
                                # face_quadrature_weights = face.get_quadrature_weights(
                                #     problem.finite_element.construction_integration_order,
                                #     quadrature_type=problem.quadrature_type,
                                # )
                                face_quadrature_size = face.get_quadrature_size(
                                    problem.finite_element.computation_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_points = face.get_quadrature_points(
                                    problem.finite_element.computation_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                face_quadrature_weights = face.get_quadrature_weights(
                                    problem.finite_element.computation_integration_order,
                                    quadrature_type=problem.quadrature_type,
                                )
                                for _qf in range(face_quadrature_size):
                                    # _h_f = element.faces[f_local].shape.diameter
                                    # _x_f = element.faces[f_local].shape.centroid
                                    _x_q_f = face_quadrature_points[:, _qf]
                                    _w_q_f = face_quadrature_weights[_qf]
                                    _s_q_f = (face_rot @ _x_q_f)[:-1]
                                    _s_f = (face_rot @ x_f)[:-1]
                                    bdf_proj = (face_rot @ bdf)[:-1]
                                    bdf_proj = face.get_face_bounding_box()
                                    # for qf in range(len(element.faces[f_local].quadrature_weights)):
                                    #     # _h_f = element.faces[f_local].shape.diameter
                                    #     # _x_f = element.faces[f_local].shape.centroid
                                    #     _x_q_f = face_quadrature_points[:, _qf]
                                    #     _w_q_f = face_quadrature_weights[_qf]
                                    #     _s_q_f = (element.faces[f_local].mapping_matrix @ _x_q_f)[:-1]
                                    #     _s_f = (element.faces[f_local].mapping_matrix @ _x_f)[:-1]
                                    # v = problem.finite_element.face_basis_k.evaluate_function(
                                    #     _x_q_f,
                                    #     element.faces[f_local].shape.centroid,
                                    #     element.faces[f_local].shape.diameter,
                                    # )
                                    # v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, h_f)
                                    v = problem.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, bdf_proj)
                                    # _x_q_f = element.faces[f_local].quadrature_points[:, qf]
                                    # _w_q_f = element.faces[f_local].quadrature_weights[qf]
                                    # v = problem.finite_element.face_basis_k.evaluate_function(
                                    #     _x_q_f,
                                    #     element.faces[f_local].shape.centroid,
                                    #     element.faces[f_local].shape.diameter,
                                    # )
                                    vf = _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                                    _c0 = _dx * _cl + f_local * _dx * _fk + boundary_condition.direction * _fk
                                    _c1 = _dx * _cl + f_local * _dx * _fk + (boundary_condition.direction + 1) * _fk
                                    # _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
                                    # _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
                                    element_external_forces[_c0:_c1] += vf
                # --- COMPUTE RESIDUAL AFTER VOLUMETRIC CONTRIBUTION
                element_residual = element_internal_forces - element_external_forces
                if verbose:
                    print("ELEM : {} | INTERNAL_FORCES_END : \n {}".format(_element_index, element_internal_forces))
                    print("ELEM : {} | ELEMENT_RESIDUAL : \n {}".format(_element_index, element_residual))
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
                _condtest = np.linalg.cond(m_cell_cell)
                m_cell_cell_inv = np.linalg.inv(m_cell_cell)
                if debug_mode == 0:
                    print("TANGENT MATRIX IN CONDENSATION COND :")
                    print("{}".format(np.linalg.cond(m_cell_cell)))
                # ge = m_faces_cell @ m_cell_cell_inv
                # gd = (m_faces_cell @ m_cell_cell_inv) @ m_cell_faces
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
                # if np.max(np.abs(element_internal_forces)) > external_forces_coefficient:
                #     external_forces_coefficient = np.max(np.abs(element_internal_forces))
            # --------------------------------------------------------------------------------------------------
            # RESIDUAL EVALUATION
            # --------------------------------------------------------------------------------------------------
            _STAB_PARAM_FINAL = _STAB_PARAM_FACTOR
            # print("STAB FACTOR : {:.6E} | MIN EIGVALS : {:.6E}".format(_STAB_PARAM_FINAL, _MIN_EIGVALS))
            if external_forces_coefficient == 0.0:
                external_forces_coefficient = 1.0
            residual_evaluation = np.max(np.abs(residual)) / external_forces_coefficient
            # print("ITER : {} | RES_MAX : {:.6E} | STAB FACTOR : {:.6E} | MIN EIGVALS : {:.6E}".format(str(iteration).zfill(4), residual_evaluation, _STAB_PARAM_FINAL, _MIN_EIGVALS))
            # print("ITER : {} | RES_MAX : {:.6E}".format(str(iteration).zfill(4), residual_evaluation))
            # if len(inte_res_failures) > 1 :
            #     print("INTEG_RES FAILURE @ QUAD_POINTS {}".format([inte_res_failures[iii][0] for iii in range(len(inte_res_failures))]))
                # print("WITH STRAIN VALUES {}".format([inte_res_failures[iii][1] for iii in range(len(inte_res_failures))]))
            # print("INTEG RES : {} @ QUAD POINT {}".format(integ_res, _qp))
            # print("ITER : {} | =====================================================================".format(str(iteration).zfill(4)))
            # residual_values.append(residual_evaluation)
            # if residual_evaluation < problem.tolerance:
            if residual_evaluation < problem.tolerance:
                print("++++ ITER : {} | RES_MAX : {:.6E} | CONVERGENCE".format(str(iteration).zfill(4), residual_evaluation))
                if len(inte_res_failures) > 1:
                    print("++++ INTEG_RES FAILURE @ QUAD_POINTS {}".format(
                        [inte_res_failures[iii][0] for iii in range(len(inte_res_failures))]))
                # ----------------------------------------------------------------------------------------------
                # UPDATE INTERNAL VARIABLES
                # ----------------------------------------------------------------------------------------------
                lagrange_parameter_buffer = np.max(np.abs(tangent_matrix))
                mgis_bv.update(material.mat_data)
                print("+ ITERATIONS : {}".format(iteration + 1))
                file_suffix = "{}".format(time_step_index).zfill(6)
                # bc_count: int = 0
                for bc in problem.boundary_conditions:
                    if bc.boundary_type == BoundaryType.DISPLACEMENT:
                        bc.force_values.append(bc.force)
                        bc.time_values.append(time_step)
                        # forces_list[bc_count].append(bc.force)
                        # forces_list[bc_count].append(boundary_force)
                        # if time_step_index > 0:
                        #     forces_list[bc_count][-1] += forces_list[bc_count][-2]
                        # bc_count += 1
                # times_values.append(time_step)
                problem.create_vertex_res_files(problem.res_folder_path, file_suffix)
                problem.create_quadrature_points_res_files(problem.res_folder_path, file_suffix, material)
                problem.write_vertex_res_files(problem.res_folder_path, file_suffix, faces_unknown_vector)
                problem.write_quadrature_points_res_files(problem.res_folder_path, file_suffix, material, faces_unknown_vector)
                # problem.create_output(problem.res_folder_path)
                problem.fill_quadrature_stress_output(problem.res_folder_path, "CAUCHY_STRESS", time_step_index_count, time_step, material)
                problem.fill_quadrature_strain_output(problem.res_folder_path, "STRAIN", time_step_index_count, time_step, material)
                problem.fill_quadrature_displacement_output(problem.res_folder_path, "QUADRATURE_DISPLACEMENT", time_step_index_count, time_step, faces_unknown_vector)
                problem.fill_node_displacement_output(problem.res_folder_path, "NODE_DISPLACEMENT", time_step_index_count, time_step, faces_unknown_vector)
                for bc in problem.boundary_conditions:
                    problem.write_force_output(problem.res_folder_path, bc)
                # problem.close_output(problem.res_folder_path)
                time_step_index_count += 1
                faces_unknown_vector_previous_step = np.copy(faces_unknown_vector)
                for element in problem.elements:
                    element.cell_unknown_vector_backup = np.copy(element.cell_unknown_vector)
                residual_values.append(residual_evaluation)
                time_step_temp = time_step + 0.
                time_step_index += 1
                break
            elif iteration == problem.number_of_iterations - 1:
                print("++++ ITER : {} | RES_MAX : {:.6E} | SPLITTING TIME STEP".format(str(iteration).zfill(4), residual_evaluation))
                if len(inte_res_failures) > 1:
                    print("++++ INTEG_RES FAILURE @ QUAD_POINTS {}".format(
                        [inte_res_failures[iii][0] for iii in range(len(inte_res_failures))]))
                # problem.time_steps.insert(time_step_index+1, (time_step + time_step_temp)/2.)
                problem.time_steps.insert(time_step_index, (time_step + time_step_temp)/2.)
                faces_unknown_vector = np.copy(faces_unknown_vector_previous_step)
                for element in problem.elements:
                    # element.cell_unknown_vector = np.zeros((_cl * _dx,))
                    element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                # time_step_index += 1
                break
            else:
                # ----------------------------------------------------------------------------------------------
                # SOLVE SYSTEM
                # ----------------------------------------------------------------------------------------------
                print("++++ ITER : {} | RES_MAX : {:.6E} | COND : {:.6E}".format(str(iteration).zfill(4), residual_evaluation, np.linalg.cond(tangent_matrix)))
                if len(inte_res_failures) > 1:
                    print("++++ INTEG_RES FAILURE @ QUAD_POINTS {}".format(
                        [inte_res_failures[iii][0] for iii in range(len(inte_res_failures))]))
                if debug_mode == 0:
                    print("K GLOBAL COND")
                    print("{}".format(np.linalg.cond(tangent_matrix)))
                # sparse_global_matrix = csr_matrix(-tangent_matrix)
                sparse_global_matrix = csr_matrix(tangent_matrix)
                print("++++++++ SOLVING THE SYSTEM")
                solving_start_time = time.process_time()
                correction = spsolve(sparse_global_matrix, residual)
                solving_end_time = time.process_time()
                system_check = np.max(np.abs(tangent_matrix @ correction - residual))
                print("++++++++ SYSTEM SOLVED IN : {:.6E}s | SYSTEM CHECK : {:.6E}".format(solving_end_time - solving_start_time, system_check))
                # print("CORRECTION :")
                # print(correction)
                # correction = np.linalg.solve(-tangent_matrix, residual)
                faces_unknown_vector += correction
                if verbose:
                    print("R_K_RES : \n {}".format(residual - tangent_matrix @ correction))
                    print(
                        "R_K_RES_NORMALIZED : \n {}".format(
                            (residual - tangent_matrix @ correction) / external_forces_coefficient
                        )
                    )
                # ----------------------------------------------------------------------------------------------
                # DECONDENSATION
                # ----------------------------------------------------------------------------------------------
                for element in problem.elements:
                    _nf = len(element.faces)
                    # _c0_c = _dx * _cl
                    # --- DECONDENSATION - GET ELEMENT UNKNOWN CORRECTION
                    face_correction = np.zeros((_nf * _fk * _dx), dtype=real)
                    # element_correction = np.zeros((element.element_size,))
                    for _i_local, _i_global in enumerate(element.faces_indices):
                        _c0_fg = _i_global * (_fk * _dx)
                        _c1_fg = (_i_global + 1) * (_fk * _dx)
                        # _c0_fl = (_dx * _cl) + _i_local * (_fk * _dx)
                        # _c1_fl = (_dx * _cl) + (_i_local + 1) * (_fk * _dx)
                        _c0_fl = _i_local * (_fk * _dx)
                        _c1_fl = (_i_local + 1) * (_fk * _dx)
                        # element_correction[_c0_fl:_c1_fl] += correction[_c0_fg:_c1_fg]
                        face_correction[_c0_fl:_c1_fl] += correction[_c0_fg:_c1_fg]
                    # element_correction[:_c0_c] = element.m_cell_cell_inv @ (
                    #     element.v_cell - element.m_cell_faces @ element_correction[_c0_c:]
                    # )
                    cell_correction = element.m_cell_cell_inv @ (
                        element.v_cell - element.m_cell_faces @ face_correction
                    )
                    # --- ADDING CORRECTION TO CURRENT DISPLACEMENT
                    element.cell_unknown_vector += cell_correction
    # plt.plot(range(len(residual_values)), residual_values, label="residual")
    # plt.plot(
    #     range(len(residual_values)), [problem.tolerance for local_i in range(len(residual_values))], label="tolerance"
    # )
    # plt.ylabel("resiudal after convergence")
    # plt.xlabel("time step index")
    # plt.title("evolution of the residual")
    # plt.legend()
    # plt.gcf().subplots_adjust(left=0.15)
    # plt.show()
    return
