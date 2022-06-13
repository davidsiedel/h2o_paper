# from h2o.problem.problem import Problem, clean_res_dir
# from h2o.problem.material import Material
# from h2o.h2o import *
#
# from mgis import behaviour_elasticity_small_strain as mgis_bv
# from scipy.sparse.linalg import spsolve
# from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# from dataclasses import dataclass
# import sys
# np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=6,suppress=True, threshold=sys.maxsize, formatter=None)
#
# from h2o.problem.output import create_output_txt
#
# def check_quadrature_point_number_consistency(problem: Problem, material: Material):
#     qp_count: int = 0
#     qp_final: int = 0
#     for element_index, element in enumerate(problem.elements):
#         qp_count += len(element.quad_p_indices)
#         if element_index == len(problem.elements) - 1:
#             qp_final = element.quad_p_indices[-1]
#     if qp_count != problem.mesh.number_of_cell_quadrature_points_in_mesh:
#         raise ArithmeticError("the number of quadrature points is not right : qp in mesh {} | qp_count {}".format(
#             problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_count))
#     if qp_final != problem.mesh.number_of_cell_quadrature_points_in_mesh - 1:
#         raise ArithmeticError("the number of quadrature points is not right : qp in mesh {} | qp_final {}".format(
#             problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_final))
#     if qp_count != material.mat_data.number_of_integration_points:
#         raise ArithmeticError("the number of quadrature points is not right : qp in mat {} | qp_count {}".format(
#             material.mat_data.number_of_integration_points, qp_count))
#     if qp_final != material.mat_data.number_of_integration_points - 1:
#         raise ArithmeticError("the number of quadrature points is not right : qp in mat {} | qp_final {}".format(
#             material.mat_data.number_of_integration_points, qp_final))
#
# class NewtonSolver:
#     problem: Problem
#     material: Material
#     # _dx: int
#     # _fk: int
#     # _cl: int
#     constrained_system_size: int
#     system_size: int
#     tangent_matrix: ndarray
#     residual: ndarray
#     faces_unknown_vector: ndarray
#     faces_unknown_vector_previous_step: ndarray
#     external_forces_coefficient: float
#
#     def __init__(self, problem: Problem, material: Material):
#         self.problem = problem
#         self.material = material
#         self.external_forces_coefficient = 1.0
#         # self._dx = self.problem.field.field_dimension
#         # self._fk = self.problem.finite_element.face_basis_k.dimension
#         # self._cl = self.problem.finite_element.cell_basis_l.dimension
#         self.constrained_system_size, self.system_size = problem.get_total_system_size()
#         self.tangent_matrix = np.zeros((self.constrained_system_size, self.constrained_system_size), dtype=real)
#         self.residual = np.zeros((self.constrained_system_size, ), dtype=real)
#         self.faces_unknown_vector = np.zeros((self.constrained_system_size, ), dtype=real)
#         self.faces_unknown_vector_previous_step = np.zeros((self.constrained_system_size, ), dtype=real)
#
#     def write_output_file(self):
#         clean_res_dir(self.problem.res_folder_path)
#         self.problem.create_output(self.problem.res_folder_path)
#         output_file_path = os.path.join(self.problem.res_folder_path, "output.txt")
#         create_output_txt(output_file_path, self.problem, self.material)
#         with open(output_file_path, "a") as outfile:
#             outfile.write(
#                 "----------------------------------------------------------------------------------------------------\n")
#             outfile.write("****************** COMPUTATION\n")
#             outfile.write("+ SYSTEM SIZE : {}\n".format(self.constrained_system_size))
#             print("+ SYSTEM SIZE : {}".format(self.constrained_system_size))
#             qp_count: int = 0
#             qp_final: int = 0
#             # --- CHECK QUADRATURE POINTS NUMBER CONSISTENCY
#             for element_index, element in enumerate(self.problem.elements):
#                 qp_count += len(element.quad_p_indices)
#                 if element_index == len(self.problem.elements) - 1:
#                     qp_final = element.quad_p_indices[-1]
#             if qp_count != self.problem.mesh.number_of_cell_quadrature_points_in_mesh:
#                 raise ArithmeticError("the number of quadrature points is not right : qp in mesh {} | qp_count {}".format(
#                     self.problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_count))
#             if qp_final != self.problem.mesh.number_of_cell_quadrature_points_in_mesh - 1:
#                 raise ArithmeticError("the number of quadrature points is not right : qp in mesh {} | qp_final {}".format(
#                     self.problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_final))
#             print("+ NUMBER OF INTEGRATION POINTS : {}".format(self.problem.mesh.number_of_cell_quadrature_points_in_mesh))
#             outfile.write(
#                 "+ NUMBER OF INTEGRATION POINTS IN MESH : {}".format(self.problem.mesh.number_of_cell_quadrature_points_in_mesh))
#             outfile.write("\n")
#
#     def newton_step(self):
#
#
#
#
#
# def newtown_step(problem: Problem, material: Material, iteration: int, break_iteration: bool):
#     _dx: int = problem.field.field_dimension
#     _fk: int = problem.finite_element.face_basis_k.dimension
#     _cl: int = problem.finite_element.cell_basis_l.dimension
#     min_eigenvals: float = +np.inf
#     max_eigenvals: float = -np.inf
#     # --------------------------------------------------------------------------------------------------
#     # SET SYSTEM MATRIX AND VECTOR
#     # --------------------------------------------------------------------------------------------------
#     tangent_matrix: ndarray = np.zeros((_constrained_system_size, _constrained_system_size), dtype=real)
#     residual: ndarray = np.zeros((_constrained_system_size), dtype=real)
#     # --------------------------------------------------------------------------------------------------
#     # SET TIME INCREMENT
#     # --------------------------------------------------------------------------------------------------
#     if time_step_index == 0:
#         _dt: float = time_step
#     else:
#         _dt: float = time_step - problem.time_steps[time_step_index - 1]
#     # --------------------------------------------------------------------------------------------------
#     # FOR ELEMENT LOOP
#     # --------------------------------------------------------------------------------------------------
#     iter_face_constraint: int = 0
#     for boundary_condition in problem.boundary_conditions:
#         if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
#             boundary_condition.force = 0.0
#     for _element_index, element in enumerate(problem.elements):
#
# def solve_system(problem: Problem, material: Material, debug_mode: DebugMode = DebugMode.NONE):
#     clean_res_dir(problem.res_folder_path)
#     problem.create_output(problem.res_folder_path)
#     output_file_path: str = os.path.join(problem.res_folder_path, "output.txt")
#     create_output_txt(output_file_path, problem, material)
#     _dx: int = problem.field.field_dimension
#     _fk: int = problem.finite_element.face_basis_k.dimension
#     _cl: int = problem.finite_element.cell_basis_l.dimension
#     external_forces_coefficient: float = 1.0
#     # ---SET SYSTEM SIZE
#     _constrained_system_size, _system_size = problem.get_total_system_size()
#     faces_unknown_vector: ndarray = np.zeros((_constrained_system_size), dtype=real)
#     faces_unknown_vector_previous_step: ndarray = np.zeros((_constrained_system_size), dtype=real)
#     with open(output_file_path, "a") as outfile:
#         # --- INITIATING COMPUTATION
#         outfile.write(
#             "----------------------------------------------------------------------------------------------------\n")
#         outfile.write("****************** COMPUTATION")
#         outfile.write("\n")
#         outfile.write("+ SYSTEM SIZE : {}".format(_constrained_system_size))
#         outfile.write("\n")
#         print("+ SYSTEM SIZE : {}".format(_constrained_system_size))
#         # --- CHECK QUADRATURE POINTS NUMBER CONSISTENCY
#         check_quadrature_point_number_consistency(problem, material)
#         print("+ NUMBER OF INTEGRATION POINTS : {}".format(problem.mesh.number_of_cell_quadrature_points_in_mesh))
#         outfile.write(
#             "+ NUMBER OF INTEGRATION POINTS IN MESH : {}".format(problem.mesh.number_of_cell_quadrature_points_in_mesh))
#         outfile.write("\n")
#         iteration: int = 0
#         break_iteration: bool = False
#         while iteration < problem.number_of_iterations and not break_iteration:
