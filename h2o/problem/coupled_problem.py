import sys
from typing import TextIO

from h2o.mesh.gmsh.data import get_element_tag
from h2o.mesh.mesh import Mesh
from h2o.fem.element.element import Element
from h2o.geometry.shape import Shape
from h2o.fem.element.finite_element import FiniteElement
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.finite_element_field import FiniteElementField, IterationOutput
from h2o.problem.load import Load
from h2o.problem.material import Material
from h2o.field.field import Field
from h2o.h2o import *
from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


def clean_res_dir(res_folder_path: str):
    """

    Args:
        res_folder_path:
    """
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


class CoupledProblem:
    finite_element_fields: List[FiniteElementField]
    time_steps: List[float]
    res_folder_path: str

    def __init__(
            self,
            finite_element_fields: List[FiniteElementField],
            time_steps: List[float],
            res_folder_path: str
    ):
        """

        Args:
            finite_element_fields:
            time_steps:
            res_folder_path:
        """
        self.finite_element_fields = finite_element_fields
        self.time_steps = time_steps
        self.res_folder_path = res_folder_path
        clean_res_dir(self.res_folder_path)
        return

    def create_output(self, res_folder_path: str):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "w") as res_output_file:
            res_output_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
            # res_output_file.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n")
            mesh = self.finite_element_fields[0].mesh
            elements = self.finite_element_fields[0].elements
            nnodes = mesh.number_of_vertices_in_mesh + mesh.number_of_cell_quadrature_points_in_mesh
            res_output_file.write("{}\n".format(nnodes))
            # res_output_file.write("1 {} 1 {}\n".format(nnodes, nnodes))
            for v_count in range(mesh.number_of_vertices_in_mesh):
                vertex_fill = np.zeros((3,), dtype=real)
                vertex_fill[:len(mesh.vertices[:,v_count])] = mesh.vertices[:,v_count]
                res_output_file.write("{} {} {} {}\n".format(v_count + 1, vertex_fill[0], vertex_fill[1], vertex_fill[2]))
            q_count = mesh.number_of_vertices_in_mesh
            for element in elements:
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
            n_elems = nnodes + len(mesh.faces_vertices_connectivity) + len(mesh.cells_vertices_connectivity)
            res_output_file.write("{}\n".format(n_elems))
            elem_count = 1
            for v_count in range(mesh.number_of_vertices_in_mesh):
                res_output_file.write("{} 15 2 0 0 {}\n".format(elem_count, elem_count))
                elem_count += 1
            # q_count = self.mesh.number_of_vertices_in_mesh
            for element in elements:
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
            for face_connectivity, face_shape in zip(mesh.faces_vertices_connectivity, mesh.faces_shape_types):
                elem_tag = get_element_tag(face_shape)
                res_output_file.write("{} {} 2 0 0 ".format(elem_count, elem_tag))
                for i_loc, coord in enumerate(face_connectivity):
                    if i_loc != len(face_connectivity) - 1:
                        res_output_file.write("{} ".format(coord + 1))
                    else:
                        res_output_file.write("{}\n".format(coord + 1))
                elem_count += 1
            for cell_connectivity, cell_shape in zip(mesh.cells_vertices_connectivity, mesh.cells_shape_types):
                elem_tag = get_element_tag(cell_shape)
                res_output_file.write("{} {} 2 0 0 ".format(elem_count, elem_tag))
                for i_loc, coord in enumerate(cell_connectivity):
                    if i_loc != len(cell_connectivity) - 1:
                        res_output_file.write("{} ".format(coord + 1))
                    else:
                        res_output_file.write("{}\n".format(coord + 1))
                elem_count += 1
            res_output_file.write("$EndElements\n")

    def make_time_step(self):
        clean_res_dir(self.res_folder_path)
        self.create_output(self.res_folder_path)
        time_step_index: int = 0
        time_step_temp: float = self.time_steps[0]
        time_split: bool = False
        while time_step_index < len(self.time_steps):
            time_step: float = self.time_steps[time_step_index]
            print("+ TIME STEP : {} VALUE : {}".format(time_step_index, time_step))
            for finite_element_field in self.finite_element_fields:
                finite_element_field.material.set_temperature()
                finite_element_field.set_material_properties()
                finite_element_field.set_external_variables(np.zeros((1, finite_element_field.material.nq), dtype=real))
                iteration: int = 0
                break_iterations: bool = False
                while not break_iterations:
                    iteration_output = finite_element_field.make_iteration(time_step, time_step_index, iteration)
                    print(iteration_output)
                    if iteration_output == IterationOutput.INTEGRATION_FAILURE or iteration == 10:
                        time_split = True
                        break_iterations = True
                    elif iteration_output == IterationOutput.SYSTEM_SOLVED:
                        iteration += 1
                    else:
                        break_iterations = True
                        time_step_temp = time_step
                        time_step_index += 1
                if time_split:
                    break
            if time_split:
                print("++++ SPLITTING TIME STEP")
                self.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                for finite_element_field in self.finite_element_fields:
                    finite_element_field.faces_unknown_vector = np.copy(finite_element_field.faces_unknown_vector_previous_step)
                    for element in finite_element_field.elements:
                        element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                time_split = False

        return

    def make_time_step2(self):
        clean_res_dir(self.res_folder_path)
        self.create_output(self.res_folder_path)
        time_step_index: int = 0
        time_step_temp: float = self.time_steps[0]
        time_split: bool = False
        while time_step_index < len(self.time_steps):
            time_step: float = self.time_steps[time_step_index]
            print("+ TIME STEP : {} VALUE : {}".format(time_step_index, time_step))
            for finite_element_field in self.finite_element_fields:
                finite_element_field.material.set_temperature()
                iteration: int = 0
                break_iterations: bool = False
                while not break_iterations:
                    iteration_output = finite_element_field.make_iteration(time_step, time_step_index, iteration)
                    if iteration_output == IterationOutput.INTEGRATION_FAILURE or iteration == 10:
                        time_split = True
                        break_iterations = True
                    elif iteration_output == IterationOutput.SYSTEM_SOLVED:
                        iteration += 1
                    else:
                        break_iterations = True
                        time_step_temp = time_step
                        time_step_index += 1
                if time_split:
                    break
            if time_split:
                print("++++ SPLITTING TIME STEP")
                self.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                for finite_element_field in self.finite_element_fields:
                    finite_element_field.faces_unknown_vector = np.copy(finite_element_field.faces_unknown_vector_previous_step)
                    for element in finite_element_field.elements:
                        element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                time_split = False

        return