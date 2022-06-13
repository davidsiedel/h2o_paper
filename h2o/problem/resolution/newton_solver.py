from h2o.problem.problem import Problem
from h2o.problem.material import Material
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.h2o import *
from h2o.mesh.gmsh.data import get_element_tag

from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=6, suppress=True,
                    threshold=sys.maxsize, formatter=None)


def clean_res_dir(res_folder_path: str):
    print(res_folder_path)
    try:
        for filename in os.listdir(res_folder_path):
            file_path = os.path.join(res_folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
    except FileNotFoundError:
        os.mkdir(res_folder_path)


class NewtonSolver:
    problem: Problem
    material: Material
    constrained_system_size: int
    system_size: int
    tangent_matrix: ndarray
    residual: ndarray
    faces_unknown_vector: ndarray
    faces_unknown_vector_previous_step: ndarray
    external_forces_coefficient: float

    def __init__(self, problem: Problem, material: Material):
        self.problem = problem
        self.material = material
        self.external_forces_coefficient = 1.0
        self.constrained_system_size, self.system_size = problem.get_total_system_size()
        self.tangent_matrix = np.zeros((self.constrained_system_size, self.constrained_system_size), dtype=real)
        self.residual = np.zeros((self.constrained_system_size,), dtype=real)
        self.faces_unknown_vector = np.zeros((self.constrained_system_size,), dtype=real)
        self.faces_unknown_vector_previous_step = np.zeros((self.constrained_system_size,), dtype=real)
        self.output_file_path = os.path.join(problem.res_folder_path, "output.txt")
        self.check_quada()
        open(self.output_file_path, "w")
        self.initiate_output_file()
        self.create_gmsh_output()

    def check_quada(self):
        qp_count: int = 0
        qp_final: int = 0
        # --- CHECK QUADRATURE POINTS NUMBER CONSISTENCY
        for element_index, element in enumerate(self.problem.elements):
            qp_count += len(element.quad_p_indices)
            if element_index == len(self.problem.elements) - 1:
                qp_final = element.quad_p_indices[-1]
        if qp_count != self.problem.mesh.number_of_cell_quadrature_points_in_mesh:
            raise ArithmeticError(
                "the number of quadrature points is not right : qp in mesh {} | qp_count {}".format(
                    self.problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_count))
        if qp_final != self.problem.mesh.number_of_cell_quadrature_points_in_mesh - 1:
            raise ArithmeticError(
                "the number of quadrature points is not right : qp in mesh {} | qp_final {}".format(
                    self.problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_final))

    def write_in_file(self, output_file_path: str, msg: str):
        with open(output_file_path, "a") as outfile:
            outfile.write(msg + "\n")

    def write_in_output_file(self, msg: str):
        self.write_in_file(self.output_file_path, msg)
        print(msg)

    def initiate_output_file(self):
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** TIME STEPS :")
        for i, ts in enumerate(self.problem.time_steps):
            self.write_in_output_file("{:.6E}".format(ts))
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** MAXIMUM NUMBER OF ITERATIONS PER STEP :")
        self.write_in_output_file("{}".format(self.problem.number_of_iterations))
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** FIELD :")
        self.write_in_output_file("DERIVATION TYPE : {}".format(self.problem.field.derivation_type.name))
        self.write_in_output_file("FILED TYPE : {}".format(self.problem.field.field_type.name))
        self.write_in_output_file("GRAD TYPE : {}".format(self.problem.field.grad_type.name))
        self.write_in_output_file("FLUX TYPE : {}".format(self.problem.field.flux_type.name))
        self.write_in_output_file("FIELD DIMENSION : {}".format(self.problem.field.field_dimension))
        self.write_in_output_file("EUCLIDEAN DIMENSION : {}".format(self.problem.field.euclidean_dimension))
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** FINITE ELEMENT :")
        self.write_in_output_file("FINITE ELEMENT TYPE : {}".format(self.problem.finite_element.element_type.name))
        self.write_in_output_file("CONSTRUCTION INTEGRATION ORDER : {}".format(self.problem.finite_element.construction_integration_order))
        self.write_in_output_file("COMPUTATION INTEGRATION ORDER : {}".format(self.problem.finite_element.computation_integration_order))
        self.write_in_output_file("FACE BASIS K ORDER : {}".format(self.problem.finite_element.face_basis_k.polynomial_order))
        self.write_in_output_file("CELL BASIS L ORDER : {}".format(self.problem.finite_element.cell_basis_l.polynomial_order))
        self.write_in_output_file("CELL BASIS K ORDER : {}".format(self.problem.finite_element.cell_basis_k.polynomial_order))
        self.write_in_output_file("CELL BASIS R ORDER : {}".format(self.problem.finite_element.cell_basis_r.polynomial_order))
        self.write_in_output_file("FACE BASIS K DIMENSION : {}".format(self.problem.finite_element.face_basis_k.dimension))
        self.write_in_output_file("CELL BASIS L DIMENSION : {}".format(self.problem.finite_element.cell_basis_l.dimension))
        self.write_in_output_file("CELL BASIS K DIMENSION : {}".format(self.problem.finite_element.cell_basis_k.dimension))
        self.write_in_output_file("CELL BASIS R DIMENSION : {}".format(self.problem.finite_element.cell_basis_r.dimension))
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** BOUNDARY CONDITIONS :")
        for i, bc in enumerate(self.problem.boundary_conditions):
            self.write_in_output_file("++++++ BOUNDARY : {}".format(bc.boundary_name))
            self.write_in_output_file("BOUNDARY TYPE : {}".format(bc.boundary_type.name))
            self.write_in_output_file("DIRECTION : {}".format(bc.direction))
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** MATERIAL :")
        self.write_in_output_file("STABILIZATION PARAMETER : {}".format(self.material.stabilization_parameter))
        self.write_in_output_file("BEHAVIOUR NAME : {}\n".format(self.material.behaviour_name))
        self.write_in_output_file("BEHAVIOUR INTEGRATION TYPE : {}".format(self.material.integration_type))
        self.write_in_output_file("NUMBER OF QUADRATURE POINTS : {}".format(self.material.nq))
        self.write_in_output_file("LAGRANGE PARAMETER : {}".format(self.material.lagrange_parameter))
        self.write_in_output_file("TEMPERATURE (K) : {}".format(self.material.temperature))
        self.write_in_output_file("-----------------------------------------------------------------------------------")
        self.write_in_output_file("****************** COMPUTATION")
        self.write_in_output_file("+ UNKNOWN SYSTEM SIZE : {}".format(self.system_size))
        self.write_in_output_file("+ TOTAL SYSTEM SIZE : {}".format(self.constrained_system_size))
        self.write_in_output_file("+ NUMBER OF INTEGRATION POINTS IN MESH : {}".format(self.problem.mesh.number_of_cell_quadrature_points_in_mesh))

    def write_force_output(self):
        for boundary_condition in self.problem.boundary_conditions:
            res_file_path = os.path.join(self.problem.res_folder_path,
                                         "output_{}.csv".format(boundary_condition.boundary_name))
            with open(res_file_path, "w") as res_output_file:
                res_output_file.write("DISPLACEMENT,LOAD\n")
                for t, f in zip(boundary_condition.time_values, boundary_condition.force_values):
                    res_output_file.write("{:.16E},{:.16E}\n".format(t, f))

    def create_gmsh_output(self, res_file_path: str):
        with open(res_file_path, "w") as res_output_file:
            res_output_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
            # res_output_file.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n")
            nnodes = self.problem.mesh.number_of_vertices_in_mesh + self.problem.mesh.number_of_cell_quadrature_points_in_mesh
            res_output_file.write("{}\n".format(nnodes))
            # res_output_file.write("1 {} 1 {}\n".format(nnodes, nnodes))
            for v_count in range(self.problem.mesh.number_of_vertices_in_mesh):
                vertex_fill = np.zeros((3,), dtype=real)
                vertex_fill[:len(self.problem.mesh.vertices[:, v_count])] = self.problem.mesh.vertices[:, v_count]
                res_output_file.write(
                    "{} {} {} {}\n".format(v_count + 1, vertex_fill[0], vertex_fill[1], vertex_fill[2]))
            q_count = self.problem.mesh.number_of_vertices_in_mesh
            for element in self.problem.elements:
                cell_quadrature_size = element.cell.get_quadrature_size(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                for qc in range(cell_quadrature_size):
                    x_q_c = cell_quadrature_points[:, qc]
                    qp_fill = np.zeros((3,), dtype=real)
                    qp_fill[:len(x_q_c)] = x_q_c
                    res_output_file.write("{} {} {} {}\n".format(q_count + 1, qp_fill[0], qp_fill[1], qp_fill[2]))
                    q_count += 1
            res_output_file.write("$EndNodes\n")
            res_output_file.write("$Elements\n")
            n_elems = nnodes + len(self.problem.mesh.faces_vertices_connectivity) + len(
                self.problem.mesh.cells_vertices_connectivity)
            res_output_file.write("{}\n".format(n_elems))
            elem_count = 1
            for v_count in range(self.problem.mesh.number_of_vertices_in_mesh):
                res_output_file.write("{} 15 2 0 0 {}\n".format(elem_count, elem_count))
                elem_count += 1
            # q_count = self.problem.mesh.number_of_vertices_in_mesh
            for element in self.problem.elements:
                cell_quadrature_size = element.cell.get_quadrature_size(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                for qc in range(cell_quadrature_size):
                    x_q_c = cell_quadrature_points[:, qc]
                    qp_fill = np.zeros((3,), dtype=real)
                    qp_fill[:len(x_q_c)] = x_q_c
                    res_output_file.write("{} 15 2 1 1 {}\n".format(elem_count, elem_count))
                    elem_count += 1
                    # res_output_file.write("{} {} {} {}\n".format(q_count + 1, qp_fill[0], qp_fill[1], qp_fill[2]))
                    # q_count += 1
            for face_connectivity, face_shape in zip(self.problem.mesh.faces_vertices_connectivity,
                                                     self.problem.mesh.faces_shape_types):
                elem_tag = get_element_tag(face_shape)
                res_output_file.write("{} {} 2 0 0 ".format(elem_count, elem_tag))
                for i_loc, coord in enumerate(face_connectivity):
                    if i_loc != len(face_connectivity) - 1:
                        res_output_file.write("{} ".format(coord + 1))
                    else:
                        res_output_file.write("{}\n".format(coord + 1))
                elem_count += 1
            for cell_connectivity, cell_shape in zip(self.problem.mesh.cells_vertices_connectivity,
                                                     self.problem.mesh.cells_shape_types):
                elem_tag = get_element_tag(cell_shape)
                res_output_file.write("{} {} 2 0 0 ".format(elem_count, elem_tag))
                for i_loc, coord in enumerate(cell_connectivity):
                    if i_loc != len(cell_connectivity) - 1:
                        res_output_file.write("{} ".format(coord + 1))
                    else:
                        res_output_file.write("{}\n".format(coord + 1))
                elem_count += 1
            res_output_file.write("$EndElements\n")
            # res_output_file.write("$NodeData\n")

    # def create_data_output(self):
    #     clean_res_dir(self.problem.res_folder_path)
    #     output_gmsh_path = os.path.join(self.problem.res_folder_path, "output.msh")
    #     self.create_gmsh_output(output_gmsh_path)
    #     output_file_path = os.path.join(self.problem.res_folder_path, "output.txt")
    #     self.create_output_txt(output_file_path)
    #     with open(output_file_path, "a") as outfile:
    #         outfile.write(
    #             "----------------------------------------------------------------------------------------------------\n")
    #         outfile.write("****************** COMPUTATION\n")
    #         outfile.write("+ SYSTEM SIZE : {}\n".format(self.constrained_system_size))
    #         print("+ SYSTEM SIZE : {}".format(self.constrained_system_size))
    #         qp_count: int = 0
    #         qp_final: int = 0
    #         # --- CHECK QUADRATURE POINTS NUMBER CONSISTENCY
    #         for element_index, element in enumerate(self.problem.elements):
    #             qp_count += len(element.quad_p_indices)
    #             if element_index == len(self.problem.elements) - 1:
    #                 qp_final = element.quad_p_indices[-1]
    #         if qp_count != self.problem.mesh.number_of_cell_quadrature_points_in_mesh:
    #             raise ArithmeticError(
    #                 "the number of quadrature points is not right : qp in mesh {} | qp_count {}".format(
    #                     self.problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_count))
    #         if qp_final != self.problem.mesh.number_of_cell_quadrature_points_in_mesh - 1:
    #             raise ArithmeticError(
    #                 "the number of quadrature points is not right : qp in mesh {} | qp_final {}".format(
    #                     self.problem.mesh.number_of_cell_quadrature_points_in_mesh, qp_final))
    #         print("+ NUMBER OF INTEGRATION POINTS : {}".format(
    #             self.problem.mesh.number_of_cell_quadrature_points_in_mesh))
    #         outfile.write(
    #             "+ NUMBER OF INTEGRATION POINTS IN MESH : {}".format(
    #                 self.problem.mesh.number_of_cell_quadrature_points_in_mesh))
    #         outfile.write("\n")
