import sys
from typing import TextIO

import numpy as np

from h2o.mesh.gmsh.data import get_element_tag
from h2o.mesh.mesh import Mesh
from h2o.fem.element.element import Element
from h2o.geometry.shape import Shape
from h2o.fem.element.finite_element import FiniteElement
from h2o.problem.boundary_condition import BoundaryCondition
# from h2o.problem.finite_element_field import FiniteElementField, IterationOutput
from h2o.problem.displacement_finite_element_field import DisplacementFiniteElementField
from h2o.problem.damage_finite_element_field import DamageFiniteElementField
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
    displacement_finite_element_field: DisplacementFiniteElementField
    damage_finite_element_field: DamageFiniteElementField
    time_steps: List[float]
    res_folder_path: str
    max_iterations: real

    def __init__(
            self,
            displacement_finite_element_field: DisplacementFiniteElementField,
            damage_finite_element_field: DamageFiniteElementField,
            time_steps: List[float],
            res_folder_path: str,
            max_iterations: real = 10
    ):
        """

        Args:
            finite_element_fields:
            time_steps:
            res_folder_path:
        """
        self.displacement_finite_element_field = displacement_finite_element_field
        self.damage_finite_element_field = damage_finite_element_field
        self.time_steps = time_steps
        self.res_folder_path = res_folder_path
        self.max_iterations = max_iterations
        clean_res_dir(self.res_folder_path)
        return

    def create_output(self, res_folder_path: str):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "w") as res_output_file:
            res_output_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
            # res_output_file.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n")
            mesh = self.displacement_finite_element_field.mesh
            elements = self.displacement_finite_element_field.elements
            nnodes = mesh.number_of_vertices_in_mesh + mesh.number_of_cell_quadrature_points_in_mesh
            res_output_file.write("{}\n".format(nnodes))
            # res_output_file.write("1 {} 1 {}\n".format(nnodes, nnodes))
            for v_count in range(mesh.number_of_vertices_in_mesh):
                vertex_fill = np.zeros((3,), dtype=real)
                vertex_fill[:len(mesh.vertices[:, v_count])] = mesh.vertices[:, v_count]
                res_output_file.write(
                    "{} {} {} {}\n".format(v_count + 1, vertex_fill[0], vertex_fill[1], vertex_fill[2]))
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
        total_iter: int = 0
        while time_step_index < len(self.time_steps):
            time_step: float = self.time_steps[time_step_index]
            print("+ TIME STEP : {} VALUE : {}".format(time_step_index, time_step))
            iteration: int = 0
            break_iterations: bool = False
            while not break_iterations:
                # ------------------------------------------------------------------------------------------------------
                # DAMAGE
                # ------------------------------------------------------------------------------------------------------
                self.damage_finite_element_field.material.set_temperature()
                self.damage_finite_element_field.set_material_properties()
                history_table = [np.zeros((self.displacement_finite_element_field.material.nq,), dtype=real)]
                load_table = [np.zeros((self.displacement_finite_element_field.material.nq,), dtype=real)]
                history_input_data = np.zeros((self.damage_finite_element_field.material.nq,), dtype=real)
                load_input_data = np.zeros((self.damage_finite_element_field.material.nq,), dtype=real)
                qp_count: int = 0
                for elem in self.damage_finite_element_field.elements:
                    int_ord: int = elem.finite_element.computation_integration_order
                    qs = elem.cell.get_quadrature_size(int_ord)
                    quad_pnts = elem.cell.get_quadrature_points(int_ord)
                    for qc in range(qs):
                        quad_pnt = quad_pnts[:, qc]
                        alpha: float = 1.0/2.0
                        G_c: float = 1.0
                        l_0: float = 0.1
                        puls: float = 4.0
                        factor: float = alpha * G_c * (1.0 + (l_0 * np.pi * puls)**2) / (2.0 * l_0)
                        nomi: float = np.sin(puls * np.pi * quad_pnt[0])
                        deno: float = 1.0 - alpha * np.sin(puls * np.pi * quad_pnt[0])
                        # v0: float = (G_c / (2.0 * l_0)) * (1.0 / (1.0 - 0.5 * np.sin(puls * np.pi * quad_pnt[0])))
                        # v1: float = (0.5 * np.sin(puls * np.pi * quad_pnt[0]) * (1.0 + (puls * np.pi * l_0) ** 2))
                        H_v: float = factor * nomi / deno
                        # H_v: float = 0.0
                        # H_v: float = -1.0
                        # H_v: float = ((4.0 * np.pi) ** 2) * np.sin(4.0 * np.pi * quad_pnt[0])
                        history_input_data[qp_count] = H_v
                        H_v = 0.0
                        load_input_data[qp_count] = 2.0 * H_v
                        qp_count += 1
                history_table = [history_input_data]
                history_table = [self.displacement_finite_element_field.get_internal_variable_table()]
                load_table = [load_input_data]
                # history_table = [np.zeros((self.displacement_finite_element_field.material.nq,), dtype=real)]
                # print("history_table :")
                # print(history_table)
                self.damage_finite_element_field.set_external_variables(history_table)
                for local_iter in range(1):
                    damage_iteration_output = self.damage_finite_element_field.make_iteration(
                        time_step,
                        time_step_index,
                        iteration,
                        external_state_vars=load_table,
                        residual_coef=np.max(load_table[0])
                        # residual_coef=1.0
                    )
                # ------------------------------------------------------------------------------------------------------
                # DISPLACEMENT
                # ------------------------------------------------------------------------------------------------------
                self.displacement_finite_element_field.material.set_temperature()
                self.displacement_finite_element_field.set_material_properties()
                # damage_table: ndarray = [np.zeros((self.displacement_finite_element_field.material.nq,), dtype=real)]
                damage_table = [self.damage_finite_element_field.get_quadrature_field_table()]
                # print("damage :")
                # print(damage_table)
                self.displacement_finite_element_field.set_external_variables(damage_table)
                for local_iter in range(1):
                    displacement_iteration_output = self.displacement_finite_element_field.make_iteration(
                        time_step,
                        time_step_index,
                        iteration
                    )
                # ------------------------------------------------------------------------------------------------------
                # self.displacement_finite_element_field.make_convergence(total_iter, total_iter)
                # self.damage_finite_element_field.make_convergence(total_iter, total_iter)
                total_iter += 1
                if displacement_iteration_output == IterationOutput.INTEGRATION_FAILURE or iteration == self.max_iterations:
                # if damage_iteration_output == IterationOutput.INTEGRATION_FAILURE or iteration == 10:
                    time_split = True
                    break_iterations = True
                elif displacement_iteration_output == IterationOutput.SYSTEM_SOLVED:
                # elif damage_iteration_output == IterationOutput.SYSTEM_SOLVED:
                #     self.displacement_finite_element_field.make_convergence(total_iter, total_iter)
                #     self.damage_finite_element_field.make_convergence(total_iter, total_iter)
                    iteration += 1
                    # total_iter += 1
                else:
                    print("+ CONVERGENCE")
                    self.displacement_finite_element_field.make_convergence(time_step_index, time_step)
                    self.damage_finite_element_field.make_convergence(time_step_index, time_step)
                    break_iterations = True
                    time_step_temp = time_step
                    time_step_index += 1
                if time_split:
                    break
            if time_split:
                print("++++ SPLITTING TIME STEP")
                self.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                # for finite_element_field in self.finite_element_fields:
                self.displacement_finite_element_field.faces_unknown_vector = np.copy(
                    self.displacement_finite_element_field.faces_unknown_vector_previous_step)
                for element in self.displacement_finite_element_field.elements:
                    element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                time_split = False
        return

    def make_time_step_fixed_point(self,
                                   external_variable_type_d: ExternalVariable,
                                   external_variable_type_u: ExternalVariable,
                                   internal_variable_index_d: int = 0,
                                   internal_variable_index_u: int = 0
                                   ):
        clean_res_dir(self.res_folder_path)
        self.create_output(self.res_folder_path)
        self.displacement_finite_element_field.create_energy_output(self.res_folder_path, self.displacement_finite_element_field.material)
        self.damage_finite_element_field.create_energy_output(self.res_folder_path, self.damage_finite_element_field.material)
        time_step_index: int = 0
        time_step_temp: float = self.time_steps[0]
        time_split: bool = False
        total_iter: int = 0
        while time_step_index < len(self.time_steps):
            time_step: float = self.time_steps[time_step_index]
            print("+ TIME STEP : {} VALUE : {}".format(time_step_index, time_step))
            # ------ GLOBAL ITERATIONS
            iteration: int = 0
            break_iterations: bool = False
            while not break_iterations:
                # ------ MECHANICS ITERATIONS
                mechanics_iteration: int = 0
                break_mechanics_iteration: bool = False
                # ------ DAMAGE ITERATIONS
                damage_iteration: int = 0
                break_damage_iteration: bool = False
                # ------------------------------------------------------------------------------------------------------
                # DISPLACEMENT PARAMETERS ACTUALIZATION
                # ------------------------------------------------------------------------------------------------------
                self.displacement_finite_element_field.material.set_temperature()
                self.displacement_finite_element_field.set_material_properties()
                if external_variable_type_u == ExternalVariable.FIELD_VARIABLE:
                    damage_table = [self.damage_finite_element_field.get_quadrature_field_table()]
                else:
                    damage_table = [self.damage_finite_element_field.get_internal_variable_table(0)]
                self.displacement_finite_element_field.set_external_variables(damage_table)
                # ------------------------------------------------------------------------------------------------------
                # DISPLACEMENT RESIDUAL MINIMIZATION
                # ------------------------------------------------------------------------------------------------------
                while not break_mechanics_iteration:
                    displacement_iteration_output = self.displacement_finite_element_field.make_iteration(
                        time_step,
                        time_step_index,
                        mechanics_iteration,
                        solve_system=True
                    )
                    if displacement_iteration_output == IterationOutput.INTEGRATION_FAILURE or mechanics_iteration == self.displacement_finite_element_field.max_iterations:
                        time_split = True
                        break_iterations = True
                        break_mechanics_iteration = True
                    elif displacement_iteration_output == IterationOutput.SYSTEM_SOLVED:
                        mechanics_iteration += 1
                    else:
                        print("+ DISPLACEMENT CONVERGENCE")
                        # self.displacement_finite_element_field.make_convergence(time_step_index, time_step)
                        break_mechanics_iteration = True
                # ------------------------------------------------------------------------------------------------------
                # DAMAGE PARAMETERS ACTUALIZATION
                # ------------------------------------------------------------------------------------------------------
                self.damage_finite_element_field.material.set_temperature()
                self.damage_finite_element_field.set_material_properties()
                if external_variable_type_d == ExternalVariable.FIELD_VARIABLE:
                    history_table = [self.displacement_finite_element_field.get_quadrature_field_table()]
                else:
                    history_table = [self.displacement_finite_element_field.get_internal_variable_table(0)]
                self.damage_finite_element_field.set_external_variables(history_table)
                # ------------------------------------------------------------------------------------------------------
                # DAMAGE RESIDUAL MINIMIZATION
                # ------------------------------------------------------------------------------------------------------
                while not break_damage_iteration:
                    damage_iteration_output = self.damage_finite_element_field.make_iteration(
                        time_step,
                        time_step_index,
                        damage_iteration,
                        solve_system=True
                    )
                    if damage_iteration_output == IterationOutput.INTEGRATION_FAILURE or damage_iteration == self.damage_finite_element_field.max_iterations:
                        time_split = True
                        break_iterations = True
                        break_damage_iteration = True
                    elif damage_iteration_output == IterationOutput.SYSTEM_SOLVED:
                        damage_iteration += 1
                    else:
                        print("+ DAMAGE CONVERGENCE")
                        # self.damage_finite_element_field.make_convergence(time_step_index, time_step)
                        break_damage_iteration = True
                # ------------------------------------------------------------------------------------------------------
                # DISPLACEMENT PARAMETERS ACTUALIZATION
                # ------------------------------------------------------------------------------------------------------
                self.displacement_finite_element_field.material.set_temperature()
                self.displacement_finite_element_field.set_material_properties()
                if external_variable_type_u == ExternalVariable.FIELD_VARIABLE:
                    damage_table = [self.damage_finite_element_field.get_quadrature_field_table()]
                else:
                    damage_table = [self.damage_finite_element_field.get_internal_variable_table(0)]
                self.displacement_finite_element_field.set_external_variables(damage_table)
                # ------------------------------------------------------------------------------------------------------
                # DISPLACEMENT CONVERGENCE TEST
                # ------------------------------------------------------------------------------------------------------
                self.displacement_finite_element_field.write_energy_output(self.res_folder_path, iteration, self.displacement_finite_element_field.material)
                self.damage_finite_element_field.write_energy_output(self.res_folder_path, iteration, self.damage_finite_element_field.material)
                displacement_iteration_output = self.displacement_finite_element_field.make_iteration(
                    time_step,
                    time_step_index,
                    0,
                    solve_system=False
                )
                if displacement_iteration_output == IterationOutput.INTEGRATION_FAILURE or iteration == self.max_iterations:
                    time_split = True
                    break_iterations = True
                elif displacement_iteration_output == IterationOutput.CONVERGENCE:
                    print("+ CONVERGENCE")
                    self.displacement_finite_element_field.make_convergence(time_step_index, time_step)
                    self.damage_finite_element_field.make_convergence(time_step_index, time_step)
                    break_iterations = True
                    time_step_temp = time_step
                    time_step_index += 1
                elif displacement_iteration_output == IterationOutput.RESIDUAL_EVALUATED:
                    iteration += 1
                    print("+ NO CONVERGENCE ---> ITER = {}".format(iteration))
                if time_split:
                    break
            if time_split:
                print("++++ SPLITTING TIME STEP")
                # ---
                self.time_steps.insert(time_step_index, (time_step + time_step_temp) / 2.)
                # ---
                self.displacement_finite_element_field.faces_unknown_vector = np.copy(
                    self.displacement_finite_element_field.faces_unknown_vector_previous_step)
                for element in self.displacement_finite_element_field.elements:
                    element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                # ---
                self.damage_finite_element_field.faces_unknown_vector = np.copy(
                    self.damage_finite_element_field.faces_unknown_vector_previous_step)
                for element in self.damage_finite_element_field.elements:
                    element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
                # ---
                time_split = False
        return
