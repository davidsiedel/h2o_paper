import sys
from typing import TextIO

import numpy as np

from h2o.mesh.gmsh.data import get_element_tag
from h2o.mesh.mesh import Mesh
from h2o.fem.element.element import Element
from h2o.geometry.shape import Shape
from h2o.fem.element.finite_element import FiniteElement
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.load import Load
from h2o.problem.material import Material
from h2o.field.field import Field
from h2o.h2o import *
from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class FiniteElementField:
    finite_element: FiniteElement
    field: Field
    mesh: Mesh
    boundary_conditions: List[BoundaryCondition]
    material: Material
    loads: List[Load]
    tolerance: float
    elements: List[Element]
    res_folder_path: str

    def __init__(
            self,
            mesh: Mesh,
            field: Field,
            finite_element: FiniteElement,
            boundary_conditions: List[BoundaryCondition],
            material: Material,
            material_properties: Dict[str, float] = None,
            external_variables: List[str] = None,
            loads: List[Load] = None,
            quadrature_type: QuadratureType = QuadratureType.GAUSS,
            tolerance: float = 1.0e-6,
            res_folder_path: str = None
    ):
        """

        Args:
            mesh_file_path:
            field:
            finite_element:
            boundary_conditions:
            material:
            loads:
            quadrature_type:
            tolerance:
            res_folder_path:
        """
        self.res_folder_path = res_folder_path
        self.finite_element = finite_element
        self.field = field
        _io = self.finite_element.k_order + self.finite_element.k_order
        self.mesh = mesh
        self.boundary_conditions = boundary_conditions
        self.material = material
        self.material_properties = material_properties
        self.external_variables = external_variables
        self.loads = loads
        self.tolerance = tolerance
        self.quadrature_type = quadrature_type
        # ------ build elements
        self.elements = self.get_elements()
        self.constrained_system_size, self.system_size = self.get_system_size()
        self.faces_unknown_vector: ndarray = np.zeros((self.constrained_system_size,), dtype=real)
        self.faces_unknown_vector_previous_step: ndarray = np.zeros((self.constrained_system_size,), dtype=real)
        self.tangent_matrix: ndarray = np.zeros((self.constrained_system_size, self.constrained_system_size),
                                                dtype=real)
        self.residual: ndarray = np.zeros((self.constrained_system_size,), dtype=real)
        self.external_forces_coefficient: real = 0.0
        self.iter_face_constraint = 0
        return

    def stress_from_LargeStrain2D(self, material: Material, qp: int) -> ndarray:
        F = np.zeros((3, 3), dtype=real)
        F[0, 0] = material.mat_data.s1.gradients[qp, 0]
        F[1, 1] = material.mat_data.s1.gradients[qp, 1]
        F[2, 2] = material.mat_data.s1.gradients[qp, 2]
        F[0, 1] = material.mat_data.s1.gradients[qp, 3]
        F[1, 0] = material.mat_data.s1.gradients[qp, 4]
        PK = np.zeros((3, 3), dtype=real)
        PK[0, 0] = material.mat_data.s1.thermodynamic_forces[qp, 0]
        PK[1, 1] = material.mat_data.s1.thermodynamic_forces[qp, 1]
        PK[2, 2] = material.mat_data.s1.thermodynamic_forces[qp, 2]
        PK[0, 1] = material.mat_data.s1.thermodynamic_forces[qp, 3]
        PK[1, 0] = material.mat_data.s1.thermodynamic_forces[qp, 4]
        J = np.linalg.det(F)
        sig = (1.0 / J) * PK @ F.T
        sig_vect = np.zeros((9,), dtype=real)
        sig_vect[0] = sig[0, 0]
        sig_vect[1] = sig[0, 1]
        sig_vect[2] = sig[0, 2]
        sig_vect[3] = sig[1, 0]
        sig_vect[4] = sig[1, 1]
        sig_vect[5] = sig[1, 2]
        sig_vect[6] = sig[2, 0]
        sig_vect[7] = sig[2, 1]
        sig_vect[8] = sig[2, 2]
        return sig_vect

    def stress_from_LargeStrain3D(self, material: Material, qp: int) -> ndarray:
        F = np.zeros((3, 3), dtype=real)
        F[0, 0] = material.mat_data.s1.gradients[qp, 0]
        F[1, 1] = material.mat_data.s1.gradients[qp, 1]
        F[2, 2] = material.mat_data.s1.gradients[qp, 2]
        F[0, 1] = material.mat_data.s1.gradients[qp, 3]
        F[1, 0] = material.mat_data.s1.gradients[qp, 4]
        F[0, 2] = material.mat_data.s1.gradients[qp, 5]
        F[2, 0] = material.mat_data.s1.gradients[qp, 6]
        F[1, 2] = material.mat_data.s1.gradients[qp, 7]
        F[2, 1] = material.mat_data.s1.gradients[qp, 8]
        PK = np.zeros((3, 3), dtype=real)
        PK[0, 0] = material.mat_data.s1.thermodynamic_forces[qp, 0]
        PK[1, 1] = material.mat_data.s1.thermodynamic_forces[qp, 1]
        PK[2, 2] = material.mat_data.s1.thermodynamic_forces[qp, 2]
        PK[0, 1] = material.mat_data.s1.thermodynamic_forces[qp, 3]
        PK[1, 0] = material.mat_data.s1.thermodynamic_forces[qp, 4]
        PK[0, 2] = material.mat_data.s1.thermodynamic_forces[qp, 5]
        PK[2, 0] = material.mat_data.s1.thermodynamic_forces[qp, 6]
        PK[1, 2] = material.mat_data.s1.thermodynamic_forces[qp, 7]
        PK[2, 1] = material.mat_data.s1.thermodynamic_forces[qp, 8]
        J = np.linalg.det(F)
        sig = (1.0 / J) * PK @ F.T
        sig_vect = np.zeros((9,), dtype=real)
        sig_vect[0] = sig[0, 0]
        sig_vect[1] = sig[0, 1]
        sig_vect[2] = sig[0, 2]
        sig_vect[3] = sig[1, 0]
        sig_vect[4] = sig[1, 1]
        sig_vect[5] = sig[1, 2]
        sig_vect[6] = sig[2, 0]
        sig_vect[7] = sig[2, 1]
        sig_vect[8] = sig[2, 2]
        return sig_vect

    def stress_from_SmallStrain3D(self, material: Material, qp: int) -> ndarray:
        sig_vect = np.zeros((9,), dtype=real)
        sig_vect[0] = material.mat_data.s1.thermodynamic_forces[qp, 0]
        sig_vect[4] = material.mat_data.s1.thermodynamic_forces[qp, 1]
        sig_vect[8] = material.mat_data.s1.thermodynamic_forces[qp, 2]
        sig_vect[1] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 3]
        sig_vect[3] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 3]
        sig_vect[2] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 4]
        sig_vect[6] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 4]
        sig_vect[5] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 5]
        sig_vect[7] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 5]
        return sig_vect

    def stress_from_SmallStrain2D(self, material: Material, qp: int) -> ndarray:
        sig_vect = np.zeros((9,), dtype=real)
        sig_vect[0] = material.mat_data.s1.thermodynamic_forces[qp, 0]
        sig_vect[4] = material.mat_data.s1.thermodynamic_forces[qp, 1]
        sig_vect[8] = material.mat_data.s1.thermodynamic_forces[qp, 2]
        sig_vect[1] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 3]
        sig_vect[3] = (1. / np.sqrt(2.)) * material.mat_data.s1.thermodynamic_forces[qp, 3]
        sig_vect[2] = 0.0
        sig_vect[6] = 0.0
        sig_vect[5] = 0.0
        sig_vect[7] = 0.0
        return sig_vect

    def strain_from_LargeStrain2D(self, material: Material, qp: int) -> ndarray:
        F = np.zeros((3, 3), dtype=real)
        F[0, 0] = material.mat_data.s1.gradients[qp, 0]
        F[1, 1] = material.mat_data.s1.gradients[qp, 1]
        F[2, 2] = material.mat_data.s1.gradients[qp, 2]
        F[0, 1] = material.mat_data.s1.gradients[qp, 3]
        F[1, 0] = material.mat_data.s1.gradients[qp, 4]
        eps_vect = np.zeros((9,), dtype=real)
        eps_vect[0] = F[0, 0]
        eps_vect[1] = F[0, 1]
        eps_vect[2] = F[0, 2]
        eps_vect[3] = F[1, 0]
        eps_vect[4] = F[1, 1]
        eps_vect[5] = F[1, 2]
        eps_vect[6] = F[2, 0]
        eps_vect[7] = F[2, 1]
        eps_vect[8] = F[2, 2]
        return eps_vect

    def strain_from_LargeStrain3D(self, material: Material, qp: int) -> ndarray:
        F = np.zeros((3, 3), dtype=real)
        F[0, 0] = material.mat_data.s1.gradients[qp, 0]
        F[1, 1] = material.mat_data.s1.gradients[qp, 1]
        F[2, 2] = material.mat_data.s1.gradients[qp, 2]
        F[0, 1] = material.mat_data.s1.gradients[qp, 3]
        F[1, 0] = material.mat_data.s1.gradients[qp, 4]
        F[0, 2] = material.mat_data.s1.gradients[qp, 5]
        F[2, 0] = material.mat_data.s1.gradients[qp, 6]
        F[1, 2] = material.mat_data.s1.gradients[qp, 7]
        F[2, 1] = material.mat_data.s1.gradients[qp, 8]
        eps_vect = np.zeros((9,), dtype=real)
        eps_vect[0] = F[0, 0]
        eps_vect[1] = F[0, 1]
        eps_vect[2] = F[0, 2]
        eps_vect[3] = F[1, 0]
        eps_vect[4] = F[1, 1]
        eps_vect[5] = F[1, 2]
        eps_vect[6] = F[2, 0]
        eps_vect[7] = F[2, 1]
        eps_vect[8] = F[2, 2]
        return eps_vect

    def strain_from_SmallStrain3D(self, material: Material, qp: int) -> ndarray:
        eps_vect = np.zeros((9,), dtype=real)
        eps_vect[0] = material.mat_data.s1.gradients[qp, 0]
        eps_vect[4] = material.mat_data.s1.gradients[qp, 1]
        eps_vect[8] = material.mat_data.s1.gradients[qp, 2]
        eps_vect[1] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 3]
        eps_vect[3] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 3]
        eps_vect[2] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 4]
        eps_vect[6] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 4]
        eps_vect[5] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 5]
        eps_vect[7] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 5]
        return eps_vect

    def strain_from_SmallStrain2D(self, material: Material, qp: int) -> ndarray:
        eps_vect = np.zeros((9,), dtype=real)
        eps_vect[0] = material.mat_data.s1.gradients[qp, 0]
        eps_vect[4] = material.mat_data.s1.gradients[qp, 1]
        eps_vect[8] = material.mat_data.s1.gradients[qp, 2]
        eps_vect[1] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 3]
        eps_vect[3] = (1. / np.sqrt(2.)) * material.mat_data.s1.gradients[qp, 3]
        eps_vect[2] = 0.0
        eps_vect[6] = 0.0
        eps_vect[5] = 0.0
        eps_vect[7] = 0.0
        return eps_vect

    def write_force_output(self, res_folder_path: str, boundary_condition: BoundaryCondition):
        """

        Args:
            res_folder_path:
            boundary_condition:
        """
        res_file_path = os.path.join(res_folder_path, "output_{}_{}.csv".format(boundary_condition.boundary_name,
                                                                                boundary_condition.direction))
        with open(res_file_path, "w") as res_output_file:
            res_output_file.write("DISPLACEMENT,LOAD\n")
            for t, f in zip(boundary_condition.time_values, boundary_condition.force_values):
                res_output_file.write("{:.16E},{:.16E}\n".format(t, f))
        return

    def fill_quadrature_stress_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                      time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.gradients[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(9))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(9):
                    if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
                        if self.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
                                                     FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS,
                                                     FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.stress_from_LargeStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_LARGE_STRAIN:
                            sig_vect = self.stress_from_LargeStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
                        if self.field.field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
                                                     FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS,
                                                     FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.stress_from_SmallStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_SMALL_STRAIN:
                            sig_vect = self.stress_from_SmallStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    if g_dir != 9 - 1:
                        res_output_file.write("{} ".format(stress_component))
                    else:
                        res_output_file.write("{}\n".format(stress_component))
            res_output_file.write("$EndNodeData\n")
        return

    def fill_quadrature_flux_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                      time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.gradients[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.material.mat_data.s1.thermodynamic_forces[0].shape[0]))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(self.material.mat_data.s1.thermodynamic_forces[0].shape[0]):
                    sig_vect = self.material.mat_data.s1.thermodynamic_forces[qp]
                    stress_component = sig_vect[g_dir]
                    if g_dir != 9 - 1:
                        res_output_file.write("{} ".format(stress_component))
                    else:
                        res_output_file.write("{}\n".format(stress_component))
            res_output_file.write("$EndNodeData\n")
        return

    def fill_quadrature_internal_variables_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                                  time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.internal_state_variables[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(field_size))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(field_size):
                    internal_variable = material.mat_data.s1.internal_state_variables[qp][g_dir]
                    if g_dir != field_size - 1:
                        res_output_file.write("{} ".format(internal_variable))
                    else:
                        res_output_file.write("{}\n".format(internal_variable))
            res_output_file.write("$EndNodeData\n")
        return

    def fill_quadrature_strain_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                      time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            # res_output_file.write("0\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.gradients[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(9))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(9):
                    if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
                        if self.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
                                                     FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS,
                                                     FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.strain_from_LargeStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_LARGE_STRAIN:
                            sig_vect = self.strain_from_LargeStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
                        if self.field.field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
                                                     FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS,
                                                     FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.strain_from_SmallStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_SMALL_STRAIN:
                            sig_vect = self.strain_from_SmallStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.SCALAR_PLANE:
                            sig_vect = self.strain_from_SmallStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    if g_dir != 9 - 1:
                        res_output_file.write("{} ".format(stress_component))
                    else:
                        res_output_file.write("{}\n".format(stress_component))
            res_output_file.write("$EndNodeData\n")
        return

    def fill_quadrature_grad_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                      time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.gradients[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.material.mat_data.s1.gradients[0].shape[0]))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(self.material.mat_data.s1.gradients[0].shape[0]):
                    sig_vect = self.material.mat_data.s1.gradients[qp]
                    stress_component = sig_vect[g_dir]
                    if g_dir != 9 - 1:
                        res_output_file.write("{} ".format(stress_component))
                    else:
                        res_output_file.write("{}\n".format(stress_component))
            res_output_file.write("$EndNodeData\n")
        return

    def fill_quadrature_displacement_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                            time_step_value: float, faces_unknown_vector: ndarray):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            # res_output_file.write("0\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(3))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            qp_point_count: int = self.mesh.number_of_vertices_in_mesh
            for elem_index, element in enumerate(self.elements):
                cell_quadrature_size = element.cell.get_quadrature_size(
                    self.finite_element.computation_integration_order, quadrature_type=self.quadrature_type
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    self.finite_element.computation_integration_order, quadrature_type=self.quadrature_type
                )
                # --- RUN OVER EACH QUADRATURE POINT
                for _qc in range(cell_quadrature_size):
                    res_output_file.write("{} ".format(qp_point_count + 1))
                    x_q_c = cell_quadrature_points[:, _qc]
                    quadp_field_value3D = np.zeros((3,), dtype=real)
                    for u_dir in range(self.field.field_dimension):
                        quad_point_field_value = element.get_cell_field_value(
                            faces_unknown_vector=faces_unknown_vector, point=x_q_c, direction=u_dir,
                        )
                        quadp_field_value3D[u_dir] = quad_point_field_value
                    for g_dir in range(3):
                        if g_dir != 3 - 1:
                            res_output_file.write("{} ".format(quadp_field_value3D[g_dir]))
                        else:
                            res_output_file.write("{}\n".format(quadp_field_value3D[g_dir]))
                    qp_point_count += 1
            res_output_file.write("$EndNodeData\n")
        return

    def fill_node_displacement_output(self, res_folder_path: str, field_label: str, time_step_index: int,
                                      time_step_value: float, faces_unknown_vector: ndarray):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            # res_output_file.write("0\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n")  # number of real tags
            res_output_file.write("{}\n".format(time_step_index))  # time step
            # res_output_file.write("{}\n".format(0)) # time step
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(3))  # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_vertices_in_mesh))
            for vertex_count in range(self.mesh.number_of_vertices_in_mesh):
                res_output_file.write("{} ".format(vertex_count + 1))
                vertex = self.mesh.vertices[:, vertex_count]
                vertex_field_value3D = np.zeros((3,), dtype=real)
                vertex_field_value = np.zeros((self.field.field_dimension,), dtype=real)
                for c, cell_vertices_connectivity in enumerate(self.mesh.cells_vertices_connectivity):
                    if vertex_count in cell_vertices_connectivity:
                        for u_dir in range(self.field.field_dimension):
                            # vertex_field_value[u_dir] += self.elements[c].get_cell_field_increment_value(
                            #     point=vertex,
                            #     direction=u_dir,
                            #     field=self.field,
                            #     finite_element=self.finite_element,
                            #     element_unknown_vector=unknown_increment,
                            # )
                            vertex_field_value[u_dir] += self.elements[c].get_cell_field_value(
                                faces_unknown_vector=faces_unknown_vector, point=vertex, direction=u_dir,
                            )
                vertex_field_value = vertex_field_value / self.mesh.vertices_weights_cell[vertex_count]
                for ddir in range(len(vertex_field_value)):
                    vertex_field_value3D[ddir] = vertex_field_value[ddir]
                for g_dir in range(3):
                    if g_dir != 3 - 1:
                        res_output_file.write("{} ".format(vertex_field_value3D[g_dir]))
                    else:
                        res_output_file.write("{}\n".format(vertex_field_value3D[g_dir]))
            res_output_file.write("$EndNodeData\n")
        return

    def get_element_quadrature_data(self, element_index: int) -> (ndarray, ndarray):
        """

        Args:
            element_index:

        Returns:

        """
        cell: Shape = self.elements[element_index].cell
        _io: int = self.finite_element.computation_integration_order
        cell_quadrature_points = cell.get_quadrature_points(
            _io, quadrature_type=self.quadrature_type
        )
        cell_quadrature_weights = cell.get_quadrature_weights(
            _io, quadrature_type=self.quadrature_type
        )
        return cell_quadrature_points, cell_quadrature_weights

    def get_face_quadrature_data(self, element_index: int, face_index: int) -> (ndarray, ndarray):
        """

        Args:
            element_index:
            face_index:

        Returns:

        """
        face: Shape = self.elements[element_index].faces[face_index]
        _io: int = self.finite_element.computation_integration_order
        face_quadrature_points = face.get_quadrature_points(
            _io,
            quadrature_type=self.quadrature_type,
        )
        face_quadrature_weights = face.get_quadrature_weights(
            _io,
            quadrature_type=self.quadrature_type,
        )
        return face_quadrature_points, face_quadrature_weights

    def make_tangent_operators(self, _qp: int):
        if self.field.field_type == FieldType.SCALAR_PLANE:
            tanop: ndarray = np.zeros((3, 3), dtype=real)
            tanop[:2, :2] = np.copy(self.material.mat_data.K[_qp, :4]).reshape((2, 2))
            tanop[2, 2] = np.copy(self.material.mat_data.K[_qp, 4])
            # print(tanop)
            return tanop
        else:
            return np.copy(self.material.mat_data.K[_qp])

    def get_element_internal_forces(self, element_index: int) -> (bool, ndarray, ndarray):
        """

        Args:
            element_index:

        Returns:

        """
        break_iteration: bool = False
        element: Element = self.elements[element_index]
        cell_quadrature_points, cell_quadrature_weights = self.get_element_quadrature_data(element_index)
        cell_quadrature_size: int = len(cell_quadrature_weights)
        element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
        element_internal_forces = np.zeros((element.element_size,), dtype=real)
        for _qc in range(cell_quadrature_size):
            _qp = element.quad_p_indices[_qc]
            _w_q_c = cell_quadrature_weights[_qc]
            _x_q_c = cell_quadrature_points[:, _qc]
            # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
            transformation_gradient = element.get_transformation_gradient(self.faces_unknown_vector, _qc)
            self.material.mat_data.s1.gradients[_qp] = transformation_gradient
            # --- INTEGRATE BEHAVIOUR LAW
            integ_res = mgis_bv.integrate(self.material.mat_data, self.material.integration_type, 0, _qp, (_qp + 1))
            if integ_res != 1:
                print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(element_index, _qp))
                break_iteration = True
                break
            else:
                # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                tanop = self.make_tangent_operators(_qp)
                # element_stiffness_matrix += _w_q_c * (
                #         element.gradients_operators[_qc].T @ self.material.mat_data.K[_qp] @
                #         element.gradients_operators[_qc]
                # )
                element_stiffness_matrix += _w_q_c * (element.operators[_qc].T @ tanop @ element.operators[_qc])
                # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
                # element_internal_forces += _w_q_c * (
                #         element.gradients_operators[_qc].T @ self.material.mat_data.s1.thermodynamic_forces[_qp]
                # )
                element_internal_forces += _w_q_c * (
                        element.operators[_qc].T @ self.material.mat_data.s1.thermodynamic_forces[_qp]
                )
        # --- STAB PARAMETER CHANGE
        stab_param = self.material.stabilization_parameter
        np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=None, suppress=True,
                            threshold=sys.maxsize, formatter=None)
        # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
        element_stiffness_matrix += stab_param * element.stabilization_operator
        # --- ADDING STABILIZATION CONTRIBUTION AT THE ELEMENT LEVEL
        element_internal_forces += (
                stab_param
                * element.stabilization_operator
                @ element.get_element_unknown_vector(self.faces_unknown_vector)
        )
        # _dx: int = self.field.field_dimension
        # _fk: int = self.finite_element.face_basis_k.dimension
        # _cl: int = self.finite_element.cell_basis_l.dimension
        # # element: Element = self.elements[element_index]
        # _c0_c: int = _dx * _cl
        # np.linalg.inv(element_stiffness_matrix[:_c0_c,:_c0_c])
        return break_iteration, element_stiffness_matrix, element_internal_forces

    def get_element_external_volumic_forces(self, element_index: int, element_external_forces: ndarray,
                                            time_step: float):
        element: Element = self.elements[element_index]
        cell_quadrature_points, cell_quadrature_weights = self.get_element_quadrature_data(element_index)
        cell_quadrature_size: int = len(cell_quadrature_weights)
        x_c: ndarray = element.cell.get_centroid()
        bdc: ndarray = element.cell.get_bounding_box()
        _cl: int = self.finite_element.cell_basis_l.dimension
        for _qc in range(cell_quadrature_size):
            _w_q_c = cell_quadrature_weights[_qc]
            _x_q_c = cell_quadrature_points[:, _qc]
            v = self.finite_element.cell_basis_l.evaluate_function(_x_q_c, x_c, bdc)
            for load in self.loads:
                vl = _w_q_c * v * load.function(time_step, _x_q_c)
                _re0 = load.direction * _cl
                _re1 = (load.direction + 1) * _cl
                element_external_forces[_re0:_re1] += vl
        return

    def get_element_external_displacement_forces(self,
                                                 element_index: int,
                                                 element_internal_forces: ndarray,
                                                 boundary_condition: BoundaryCondition,
                                                 time_step: float):
        """

        Args:
            element_index:
            element_internal_forces:
            boundary_condition:
            time_step:

        Returns:

        """
        _dx: int = self.field.field_dimension
        _fk: int = self.finite_element.face_basis_k.dimension
        _cl: int = self.finite_element.cell_basis_l.dimension
        element: Element = self.elements[element_index]
        for f_local, f_global in enumerate(element.faces_indices):
            if f_global in self.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]:
                face_quadrature_points, face_quadrature_weights = self.get_face_quadrature_data(element_index, f_local)
                face_quadrature_size: int = len(face_quadrature_weights)
                _l0 = self.system_size + self.iter_face_constraint * _fk
                _l1 = self.system_size + (self.iter_face_constraint + 1) * _fk
                _c0 = _cl * _dx + (f_local * _dx * _fk) + boundary_condition.direction * _fk
                _c1 = _cl * _dx + (f_local * _dx * _fk) + (boundary_condition.direction + 1) * _fk
                _r0 = f_global * _fk * _dx + _fk * boundary_condition.direction
                _r1 = f_global * _fk * _dx + _fk * (boundary_condition.direction + 1)
                face_lagrange = self.faces_unknown_vector[_l0:_l1]
                face_displacement = self.faces_unknown_vector[_r0:_r1]
                _m_psi_psi_face = np.zeros((_fk, _fk), dtype=real)
                _v_face_imposed_displacement = np.zeros((_fk,), dtype=real)
                face = element.faces[f_local]
                x_f = face.get_centroid()
                face_rot = face.get_rotation_matrix()
                bdf_proj = face.get_face_bounding_box()
                for _qf in range(face_quadrature_size):
                    _x_q_f = face_quadrature_points[:, _qf]
                    _w_q_f = face_quadrature_weights[_qf]
                    _s_q_f = (face_rot @ _x_q_f)[:-1]
                    _s_f = (face_rot @ x_f)[:-1]
                    _psi_k = self.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, bdf_proj)
                    _m_psi_psi_face += _w_q_f * np.tensordot(_psi_k, _psi_k, axes=0)
                    v = self.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f, bdf_proj)
                    _v_face_imposed_displacement += (_w_q_f * v * boundary_condition.function(time_step, _x_q_f))
                force_item = self.material.lagrange_parameter * (np.ones(_fk) @ face_lagrange[:])
                boundary_condition.force += force_item
                _m_psi_psi_face_inv = np.linalg.inv(_m_psi_psi_face)
                imposed_face_displacement = _m_psi_psi_face_inv @ _v_face_imposed_displacement
                face_displacement_difference = face_displacement - imposed_face_displacement
                # --- LAGRANGE INTERNAL FORCES PART
                element_internal_forces[_c0:_c1] += self.material.lagrange_parameter * face_lagrange
                self.residual[_l0:_l1] -= self.material.lagrange_parameter * face_displacement_difference
                # --- LAGRANGE MATRIX PART
                self.tangent_matrix[_l0:_l1, _r0:_r1] += self.material.lagrange_parameter * np.eye(
                    _fk, dtype=real
                )
                self.tangent_matrix[_r0:_r1, _l0:_l1] += self.material.lagrange_parameter * np.eye(
                    _fk, dtype=real
                )
                # --- SET EXTERNAL FORCES COEFFICIENT
                lagrange_external_forces = (
                        self.material.lagrange_parameter * imposed_face_displacement
                )
                if np.max(np.abs(lagrange_external_forces)) > self.external_forces_coefficient:
                    self.external_forces_coefficient = np.max(np.abs(lagrange_external_forces))
                self.iter_face_constraint += 1
        return

    def get_element_external_contact_forces(self,
                                            element_index: int,
                                            element_external_forces: ndarray,
                                            boundary_condition: BoundaryCondition,
                                            time_step: float):
        """

        Args:
            element_index:
            element_external_forces:
            boundary_condition:
            external_forces_coefficient:
            time_step:
        """
        _dx: int = self.field.field_dimension
        _fk: int = self.finite_element.face_basis_k.dimension
        _cl: int = self.finite_element.cell_basis_l.dimension
        element: Element = self.elements[element_index]
        for f_local, f_global in enumerate(element.faces_indices):
            if f_global in self.mesh.faces_boundaries_connectivity[boundary_condition.boundary_name]:
                face = element.faces[f_local]
                x_f = face.get_centroid()
                face_rot = face.get_rotation_matrix()
                _io: int = self.finite_element.k_order
                _io: int = 8
                _io: int = self.finite_element.computation_integration_order
                face_quadrature_size = face.get_quadrature_size(
                    _io, quadrature_type=self.quadrature_type,
                )
                face_quadrature_points = face.get_quadrature_points(
                    _io, quadrature_type=self.quadrature_type,
                )
                face_quadrature_weights = face.get_quadrature_weights(
                    _io, quadrature_type=self.quadrature_type,
                )
                for _qf in range(face_quadrature_size):
                    _x_q_f = face_quadrature_points[:, _qf]
                    _w_q_f = face_quadrature_weights[_qf]
                    _s_q_f = (face_rot @ _x_q_f)[:-1]
                    _s_f = (face_rot @ x_f)[:-1]
                    bdf_proj = face.get_face_bounding_box()
                    v = self.finite_element.face_basis_k.evaluate_function(_s_q_f, _s_f,
                                                                           bdf_proj)
                    vf = _w_q_f * v * boundary_condition.function(time_step, _x_q_f)
                    _c0 = _dx * _cl + f_local * _dx * _fk + boundary_condition.direction * _fk
                    _c1 = _dx * _cl + f_local * _dx * _fk + (
                            boundary_condition.direction + 1) * _fk
                    element_external_forces[_c0:_c1] += vf
                if np.max(np.abs(element_external_forces)) > self.external_forces_coefficient:
                    self.external_forces_coefficient = np.max(np.abs(element_external_forces))
        return

    def make_condensation(self, element_index: int, element_residual: ndarray, element_stiffness_matrix: ndarray):
        """

        Args:
            element_index:
            element_residual:
            element_stiffness_matrix:

        Returns:

        """
        _dx: int = self.field.field_dimension
        _fk: int = self.finite_element.face_basis_k.dimension
        _cl: int = self.finite_element.cell_basis_l.dimension
        element: Element = self.elements[element_index]
        _c0_c: int = _dx * _cl
        m_cell_cell = element_stiffness_matrix[:_c0_c, :_c0_c]
        m_cell_faces = element_stiffness_matrix[:_c0_c, _c0_c:]
        m_faces_cell = element_stiffness_matrix[_c0_c:, :_c0_c]
        m_faces_faces = element_stiffness_matrix[_c0_c:, _c0_c:]
        v_cell = -element_residual[:_c0_c]
        v_faces = -element_residual[_c0_c:]
        m_cell_cell_inv = np.linalg.inv(m_cell_cell)
        k_cond = m_faces_faces - ((m_faces_cell @ m_cell_cell_inv) @ m_cell_faces)
        r_cond = v_faces - (m_faces_cell @ m_cell_cell_inv) @ v_cell
        # --- SET CONDENSATION/DECONDENSATION MATRICES
        element.m_cell_cell_inv = m_cell_cell_inv
        element.m_cell_faces = m_cell_faces
        element.v_cell = v_cell
        return k_cond, r_cond

    def make_element_assembly(self, element_index: int, k_cond: ndarray, r_cond: ndarray):
        """

        Args:
            element_index:
            k_cond:
            r_cond:

        Returns:

        """
        _dx: int = self.field.field_dimension
        _fk: int = self.finite_element.face_basis_k.dimension
        element: Element = self.elements[element_index]
        for _i_local, _i_global in enumerate(element.faces_indices):
            _rg0 = _i_global * (_fk * _dx)
            _rg1 = (_i_global + 1) * (_fk * _dx)
            _re0 = _i_local * (_fk * _dx)
            _re1 = (_i_local + 1) * (_fk * _dx)
            self.residual[_rg0:_rg1] += r_cond[_re0:_re1]
            for _j_local, _j_global in enumerate(element.faces_indices):
                _cg0 = _j_global * (_fk * _dx)
                _cg1 = (_j_global + 1) * (_fk * _dx)
                _ce0 = _j_local * (_fk * _dx)
                _ce1 = (_j_local + 1) * (_fk * _dx)
                self.tangent_matrix[_rg0:_rg1, _cg0:_cg1] += k_cond[_re0:_re1, _ce0:_ce1]
        return

    def make_decondensation(self, element_index: int, correction: ndarray):
        """

        Args:
            element_index:
            correction:
        """
        _dx: int = self.field.field_dimension
        _fk: int = self.finite_element.face_basis_k.dimension
        _cl: int = self.finite_element.cell_basis_l.dimension
        element: Element = self.elements[element_index]
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

    def make_convergence(self, time_step_index: int, time_step: float):
        """

        Args:
            time_step_index:
            time_step:
        """
        mgis_bv.update(self.material.mat_data)
        for bc in self.boundary_conditions:
            if bc.boundary_type == BoundaryType.DISPLACEMENT:
                bc.force_values.append(bc.force)
                bc.time_values.append(time_step)
        if not self.material.behaviour_name in ["Elasticity", "Signorini", "PhaseFieldDamage"]:
            self.fill_quadrature_internal_variables_output(
                self.res_folder_path, "INTERNAL_VARIABLES_" + self.field.label, time_step_index, time_step,
                self.material
            )
        if self.field.field_type == FieldType.SCALAR_PLANE:
            self.fill_quadrature_flux_output(self.res_folder_path,
                                               "FLUX_" + self.field.label,
                                               time_step_index,
                                               time_step,
                                               self.material)
            self.fill_quadrature_grad_output(self.res_folder_path,
                                               "GRAD_" + self.field.label,
                                               time_step_index,
                                               time_step,
                                               self.material)
        else:
            self.fill_quadrature_stress_output(self.res_folder_path,
                                               "FLUX_" + self.field.label,
                                               time_step_index,
                                               time_step,
                                               self.material)
            self.fill_quadrature_strain_output(self.res_folder_path,
                                               "GRADIENT_" + self.field.label,
                                               time_step_index,
                                               time_step,
                                               self.material)
        self.fill_quadrature_displacement_output(self.res_folder_path,
                                                 "QUADRATURE_FIELD_" + self.field.label,
                                                 time_step_index,
                                                 time_step,
                                                 self.faces_unknown_vector)
        self.fill_node_displacement_output(self.res_folder_path,
                                           "NODE_FIELD_" + self.field.label,
                                           time_step_index,
                                           time_step,
                                           self.faces_unknown_vector)
        for bc in self.boundary_conditions:
            self.write_force_output(self.res_folder_path, bc)
        self.faces_unknown_vector_previous_step = np.copy(self.faces_unknown_vector)
        for element in self.elements:
            element.cell_unknown_vector_backup = np.copy(element.cell_unknown_vector)
        return

    def make_maximum_iterations_reached(self):
        """

        Returns:

        """
        self.faces_unknown_vector = np.copy(self.faces_unknown_vector_previous_step)
        for element in self.elements:
            element.cell_unknown_vector = np.copy(element.cell_unknown_vector_backup)
        return

    def set_external_variables(self, data: ndarray):
        """

        Args:
            data:
        """
        for i, external_variable in enumerate(self.external_variables):
            mgis_bv.setExternalStateVariable(self.material.mat_data.s0,
                                             external_variable,
                                             data[i],
                                             mgis_bv.MaterialStateManagerStorageMode.LOCAL_STORAGE)
            mgis_bv.setExternalStateVariable(self.material.mat_data.s1,
                                             external_variable,
                                             data[i],
                                             mgis_bv.MaterialStateManagerStorageMode.LOCAL_STORAGE)

    def set_material_properties(self):
        """

        Args:
            data:
        """
        for key, val in self.material_properties.items():
            mgis_bv.setMaterialProperty(self.material.mat_data.s0,
                                        key,
                                        val)
            mgis_bv.setMaterialProperty(self.material.mat_data.s1,
                                        key,
                                        val)

    def make_iteration(self, time_step: real, time_step_index: int, iteration: int):
        """

        Args:
            time_step:
            time_step_index:
            iteration:
            outfile:

        Returns:

        """
        _dx: int = self.field.field_dimension
        _fk: int = self.finite_element.face_basis_k.dimension
        _cl: int = self.finite_element.cell_basis_l.dimension
        self.tangent_matrix: ndarray = np.zeros((self.constrained_system_size, self.constrained_system_size),
                                                dtype=float)
        self.residual: ndarray = np.zeros((self.constrained_system_size,), dtype=real)
        self.external_forces_coefficient = 0.0
        self.iter_face_constraint: int = 0
        for boundary_condition in self.boundary_conditions:
            if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                boundary_condition.force = 0.0
        for _element_index, element in enumerate(self.elements):
            # --- INITIALIZE MATRIX AND VECTORS
            element_external_forces = np.zeros((element.element_size,), dtype=real)
            break_iteration, element_stiffness_matrix, element_internal_forces = self.get_element_internal_forces(
                _element_index)
            self.get_element_external_volumic_forces(_element_index, element_external_forces, time_step)
            if not break_iteration:
                # --- BOUNDARY CONDITIONS
                for boundary_condition in self.boundary_conditions:
                    # --- DISPLACEMENT CONDITIONS
                    if boundary_condition.boundary_type == BoundaryType.DISPLACEMENT:
                        self.get_element_external_displacement_forces(
                            _element_index,
                            element_internal_forces,
                            boundary_condition,
                            time_step)
                    elif boundary_condition.boundary_type == BoundaryType.PRESSURE:
                        self.get_element_external_contact_forces(
                            _element_index,
                            element_external_forces,
                            boundary_condition,
                            time_step)
                element_residual = element_internal_forces - element_external_forces
                k_cond, r_cond = self.make_condensation(_element_index, element_residual, element_stiffness_matrix)
                # --- ASSEMBLY
                self.make_element_assembly(_element_index, k_cond, r_cond)
            else:
                return IterationOutput.INTEGRATION_FAILURE
        print("++++ DONE ITERATION OVER ELEMENTS")
        # --------------------------------------------------------------------------------------------------
        # RESIDUAL EVALUATION
        # --------------------------------------------------------------------------------------------------
        if self.external_forces_coefficient == 0.0:
            self.external_forces_coefficient = 1.0
        print("++++ RESIDUAL EVALUATION")
        residual_evaluation = np.max(np.abs(self.residual)) / self.external_forces_coefficient
        if residual_evaluation < self.tolerance:
            print("++++ ITER : {} | RES_MAX : {:.6E} | TOLERANCE {:.6E} | CONVERGENCE".format(
                str(iteration).zfill(4), residual_evaluation, self.tolerance))
            # --- UPDATE INTERNAL VARIABLES
            self.make_convergence(time_step_index, time_step)
            print("+ ITERATIONS : {}".format(iteration + 1))
            return IterationOutput.CONVERGENCE
        else:
            # --- SOLVE SYSTEM
            print("++++ ITER : {} | RES_MAX : {:.6E}".format(
                str(iteration).zfill(4), residual_evaluation))
            sparse_global_matrix = csr_matrix(self.tangent_matrix)
            correction = spsolve(sparse_global_matrix, self.residual)
            self.faces_unknown_vector += correction
            # --- DECONDENSATION
            for _element_index, element in enumerate(self.elements):
                self.make_decondensation(_element_index, correction)
            return IterationOutput.SYSTEM_SOLVED

    def get_system_size(self) -> (int, int):
        """

        Returns:

        """
        constrained_faces = 0
        constrained_constants = 0
        for key, val in self.mesh.faces_boundaries_connectivity.items():
            for bc in self.boundary_conditions:
                if key == bc.boundary_name and bc.boundary_type == BoundaryType.DISPLACEMENT:
                    constrained_faces += len(val)
                elif key == bc.boundary_name and bc.boundary_type == BoundaryType.SLIDE:
                    constrained_constants += len(val)
        constrained_faces_matrix_size = constrained_faces * self.finite_element.face_basis_k.dimension
        constrained_const_matrix_size = constrained_constants
        lagrange_system_size = constrained_faces_matrix_size + constrained_const_matrix_size
        system_size = self.mesh.number_of_faces_in_mesh * self.finite_element.face_basis_k.dimension * self.field.field_dimension
        constrained_system_size = system_size + lagrange_system_size
        return constrained_system_size, system_size

    def get_elements(self):
        """

        Returns:

        """
        mean_time = 0.
        elements = []
        _fk = self.finite_element.face_basis_k.dimension
        _dx = self.field.field_dimension
        _cl = self.finite_element.cell_basis_l.dimension
        qp_count = 0
        for cell_index in range(self.mesh.number_of_cells_in_mesh):
            cell_vertices_connectivity = self.mesh.cells_vertices_connectivity[cell_index]
            cell_faces_connectivity = self.mesh.cells_faces_connectivity[cell_index]
            cell_ordering = self.mesh.cells_ordering[cell_index]
            cell_shape_type = self.mesh.cells_shape_types[cell_index]
            cell_vertices = self.mesh.vertices[:, cell_vertices_connectivity]
            # element_cell = Cell(cell_shape_type, cell_vertices, integration_order, quadrature_type=quadrature_type)
            element_cell = Shape(cell_shape_type, cell_vertices, connectivity=cell_ordering)
            cell_quadrature_size = element_cell.get_quadrature_size(self.finite_element.computation_integration_order)
            element_faces = []
            element_faces_indices = []
            for global_face_index in cell_faces_connectivity:
                element_faces_indices.append(global_face_index)
                face_vertices_indices = self.mesh.faces_vertices_connectivity[global_face_index]
                face_vertices = self.mesh.vertices[:, face_vertices_indices]
                face_shape_type = self.mesh.faces_shape_types[global_face_index]
                # print(global_face_index)
                # print(face_vertices)
                # print(cell_vertices)
                face = Shape(face_shape_type, face_vertices)
                element_faces.append(face)
            # print(cell_index)
            # print(cell_vertices)
            startime = time.process_time()
            quad_p_indices = [qp_count + i for i in range(cell_quadrature_size)]
            element = Element(
                self.field,
                self.finite_element,
                element_cell,
                element_faces,
                element_faces_indices,
                quad_p_indices
            )
            qp_count += cell_quadrature_size
            endtime = time.process_time()
            mean_time += endtime - startime
            elements.append(element)
            del element_cell
            del element_faces
        mean_time /= len(elements)
        print("ELEMNT BUILD TIME : {} | ORDERS : {}, {}".format(mean_time,
                                                                self.finite_element.face_basis_k.polynomial_order,
                                                                self.finite_element.cell_basis_l.polynomial_order))
        return elements
