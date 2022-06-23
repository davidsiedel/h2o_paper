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


def clean_res_dir(res_folder_path: str):
    """

    """
    # res_folder = os.path.join(get_project_path(), "res")
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


class Problem:
    finite_element: FiniteElement
    field: Field
    mesh: Mesh
    boundary_conditions: List[BoundaryCondition]
    loads: List[Load]
    time_steps: List[float]
    number_of_iterations: int
    tolerance: float
    elements: List[Element]
    res_folder_path: str

    def __init__(
        self,
        mesh_file_path: str,
        field: Field,
        time_steps: List[float],
        iterations: int,
        finite_element: FiniteElement,
        boundary_conditions: List[BoundaryCondition],
        loads: List[Load] = None,
        quadrature_type: QuadratureType = QuadratureType.GAUSS,
        tolerance: float = 1.0e-6,
        res_folder_path: str = None
    ):
        """

        Args:
            mesh_file_path:
            field:
            time_steps:
            iterations:
            finite_element:
            boundary_conditions:
            loads:
            quadrature_type:
            tolerance:
        """
        self.res_folder_path = res_folder_path
        self.finite_element = finite_element
        self.field = field
        self.mesh_file_path = mesh_file_path
        # self.mesh = Mesh(mesh_file_path=mesh_file_path, integration_order=finite_element.construction_integration_order)
        _io = self.finite_element.k_order + self.finite_element.k_order
        self.mesh = Mesh(mesh_file_path=mesh_file_path, integration_order=finite_element.computation_integration_order)
        # self.mesh = Mesh(mesh_file_path=mesh_file_path, integration_order=_io)
        self.__check_loads(loads)
        self.__check_boundary_conditions(boundary_conditions)
        self.boundary_conditions = boundary_conditions
        self.loads = loads
        self.time_steps = list(time_steps)
        self.number_of_iterations = iterations
        self.tolerance = tolerance
        self.quadrature_type = quadrature_type
        # ------ build elements
        self.elements = self.get_elements()
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

    def count_forces(self) -> int:
        count: int = 0
        for bc in self.boundary_conditions:
            if bc.boundary_type == BoundaryType.DISPLACEMENT:
                count += 1
        return count

    def get_cell_system_size(self) -> int:
        return self.mesh.number_of_cells_in_mesh * self.finite_element.cell_basis_l.dimension * self.field.field_dimension

    def get_total_system_size(self) -> (int, int):
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

    def write_force_output(self, res_folder_path: str, boundary_condition: BoundaryCondition):
        res_file_path = os.path.join(res_folder_path, "output_{}_{}.csv".format(boundary_condition.boundary_name, boundary_condition.direction))
        with open(res_file_path, "w") as res_output_file:
            res_output_file.write("DISPLACEMENT,LOAD\n")
            for t, f in zip(boundary_condition.time_values, boundary_condition.force_values):
                res_output_file.write("{:.16E},{:.16E}\n".format(t,f))

    def create_output(self, res_folder_path: str):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "w") as res_output_file:
            res_output_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
            # res_output_file.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n")
            nnodes = self.mesh.number_of_vertices_in_mesh + self.mesh.number_of_cell_quadrature_points_in_mesh
            res_output_file.write("{}\n".format(nnodes))
            # res_output_file.write("1 {} 1 {}\n".format(nnodes, nnodes))
            for v_count in range(self.mesh.number_of_vertices_in_mesh):
                vertex_fill = np.zeros((3,), dtype=real)
                vertex_fill[:len(self.mesh.vertices[:,v_count])] = self.mesh.vertices[:,v_count]
                res_output_file.write("{} {} {} {}\n".format(v_count + 1, vertex_fill[0], vertex_fill[1], vertex_fill[2]))
            q_count = self.mesh.number_of_vertices_in_mesh
            for element in self.elements:
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
            n_elems = nnodes + len(self.mesh.faces_vertices_connectivity) + len(self.mesh.cells_vertices_connectivity)
            res_output_file.write("{}\n".format(n_elems))
            elem_count = 1
            for v_count in range(self.mesh.number_of_vertices_in_mesh):
                res_output_file.write("{} 15 2 0 0 {}\n".format(elem_count, elem_count))
                elem_count += 1
            # q_count = self.mesh.number_of_vertices_in_mesh
            for element in self.elements:
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
            for face_connectivity, face_shape in zip(self.mesh.faces_vertices_connectivity, self.mesh.faces_shape_types):
                elem_tag = get_element_tag(face_shape)
                res_output_file.write("{} {} 2 0 0 ".format(elem_count, elem_tag))
                for i_loc, coord in enumerate(face_connectivity):
                    if i_loc != len(face_connectivity) - 1:
                        res_output_file.write("{} ".format(coord + 1))
                    else:
                        res_output_file.write("{}\n".format(coord + 1))
                elem_count += 1
            for cell_connectivity, cell_shape in zip(self.mesh.cells_vertices_connectivity, self.mesh.cells_shape_types):
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

    def fill_quadrature_stress_output(self, res_folder_path: str, field_label: str, time_step_index: int, time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n") # number of real tags
            res_output_file.write("{}\n".format(time_step_index)) # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.gradients[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(9)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(9):
                    if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
                        if self.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN, FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS, FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.stress_from_LargeStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_LARGE_STRAIN:
                            sig_vect = self.stress_from_LargeStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
                        if self.field.field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN, FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS, FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
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

    def fill_quadrature_internal_variables_output(self, res_folder_path: str, field_label: str, time_step_index: int, time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n") # number of real tags
            res_output_file.write("{}\n".format(time_step_index)) # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.internal_state_variables[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
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

    def fill_quadrature_strain_output(self, res_folder_path: str, field_label: str, time_step_index: int, time_step_value: float, material: Material):
        res_file_path = os.path.join(res_folder_path, "output.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$NodeData\n")
            res_output_file.write("1\n")
            res_output_file.write("\"{}\"\n".format(field_label))
            res_output_file.write("1\n")
            # res_output_file.write("0\n")
            res_output_file.write("{}\n".format(time_step_value))
            res_output_file.write("3\n") # number of real tags
            res_output_file.write("{}\n".format(time_step_index)) # time step
            # res_output_file.write("{}\n".format(0)) # time step
            field_size = len(material.mat_data.s1.gradients[0])
            # res_output_file.write("{}\n".format(field_size)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(9)) # dim of the field (vector 3, tensor 9, ...)
            res_output_file.write("{}\n".format(self.mesh.number_of_cell_quadrature_points_in_mesh))
            for qp in range(material.mat_data.n):
                res_output_file.write("{} ".format(qp + 1 + self.mesh.number_of_vertices_in_mesh))
                for g_dir in range(9):
                    if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
                        if self.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN, FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS, FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.strain_from_LargeStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_LARGE_STRAIN:
                            sig_vect = self.strain_from_LargeStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
                        if self.field.field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN, FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS, FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
                            sig_vect = self.strain_from_SmallStrain2D(material, qp)
                            stress_component = sig_vect[g_dir]
                        elif self.field.field_type == FieldType.DISPLACEMENT_SMALL_STRAIN:
                            sig_vect = self.strain_from_SmallStrain3D(material, qp)
                            stress_component = sig_vect[g_dir]
                    if g_dir != 9 - 1:
                        res_output_file.write("{} ".format(stress_component))
                    else:
                        res_output_file.write("{}\n".format(stress_component))
            res_output_file.write("$EndNodeData\n")

    def fill_quadrature_displacement_output(self, res_folder_path: str, field_label: str, time_step_index: int, time_step_value: float, faces_unknown_vector: ndarray):
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

    def fill_node_displacement_output(self, res_folder_path: str, field_label: str, time_step_index: int, time_step_value: float, faces_unknown_vector: ndarray):
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

    def close_output(self, res_folder_path: str):
        res_file_path = os.path.join(res_folder_path, "output0.msh")
        with open(res_file_path, "a") as res_output_file:
            res_output_file.write("$EndNodeData\n")



    def create_vertex_res_files(self, res_folder_path: str, suffix: str):
        """

        Args:
            res_folder_path:
            suffix:

        Returns:

        """
        res_file_path = os.path.join(res_folder_path, "res_vtx_{}.csv".format(suffix))
        with open(res_file_path, "w") as res_vtx_file:
        # with open(os.path.join(get_project_path(), "res/res_vtx_{}.csv".format(suffix)), "w") as res_vtx_file:
            for x_dir in range(self.field.euclidean_dimension):
                res_vtx_file.write("X_{},".format(x_dir))
            for u_dir in range(self.field.field_dimension):
                res_vtx_file.write("{}_{},".format(self.field.label, u_dir))
            res_vtx_file.write("\n")
        return

    def create_quadrature_points_res_files(self, res_folder_path: str, suffix: str, material: Material):
        """

        Args:
            res_folder_path:
            suffix:
            material:
        """
        res_file_path = os.path.join(res_folder_path, "res_qdp_{}.csv".format(suffix))
        with open(res_file_path, "w") as res_qdp_file:
        # with open(os.path.join(get_project_path(), "res/res_qdp_{}.csv".format(suffix)), "w") as res_qdp_file:
            for x_dir in range(self.field.euclidean_dimension):
                res_qdp_file.write("XQ_{},".format(x_dir))
            for u_dir in range(self.field.field_dimension):
                res_qdp_file.write("{}_{},".format(self.field.label, u_dir))
            for strain_component in range(self.field.gradient_dimension):
                res_qdp_file.write("STRAIN_{},".format(strain_component))
            for stress_component in range(self.field.gradient_dimension):
                res_qdp_file.write("STRESS_{},".format(stress_component))
            res_qdp_file.write("STRAIN_TRACE,")
            res_qdp_file.write("HYDRO_STRESS,")
            if material.behaviour_name not in ["Elasticity", "Signorini"]:
                # try:
                isv = material.mat_data.s1.internal_state_variables
                for isv_val in range(len(isv[0])):
                    res_qdp_file.write("INTERNAL_STATE_VARIABLE_{},".format(isv_val))
            # except:
            #     pass
            # stored_energies
            # ', '
            # dissipated_energies
            # ', '
            # internal_state_variables
            # '
            # for
            res_qdp_file.write("\n")

    def write_vertex_res_files(self, res_folder_path: str, suffix: str, faces_unknown_vector: ndarray):
        """

        Args:
            res_folder_path:
            suffix:
            faces_unknown_vector:

        Returns:

        """
        res_file_path = os.path.join(res_folder_path, "res_vtx_{}.csv".format(suffix))
        with open(res_file_path, "a") as res_vtx_file:
        # with open(os.path.join(get_project_path(), "res/res_vtx_{}.csv".format(suffix)), "a") as res_vtx_file:
            for vertex_count in range(self.mesh.number_of_vertices_in_mesh):
                vertex = self.mesh.vertices[:, vertex_count]
                for x_dir in range(self.field.euclidean_dimension):
                    res_vtx_file.write("{},".format(vertex[x_dir]))
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
                for u_dir in range(self.field.field_dimension):
                    res_vtx_file.write("{},".format(vertex_field_value[u_dir]))
                res_vtx_file.write("\n")
        return

    def write_quadrature_points_res_files(self, res_folder_path: str, suffix: str, material: Material, faces_unknown_vector: ndarray):
        """

        Args:
            res_folder_path:
            suffix:
            material:
            faces_unknown_vector:

        Returns:

        """
        res_file_path = os.path.join(res_folder_path, "res_qdp_{}.csv".format(suffix))
        with open(res_file_path, "a") as res_qdp_file:
        # with open(os.path.join(get_project_path(), "res/res_qdp_{}.csv".format(suffix)), "a") as res_qdp_file:
            qp = 0
            for element in self.elements:
                cell_quadrature_size = element.cell.get_quadrature_size(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                cell_quadrature_points = element.cell.get_quadrature_points(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                cell_quadrature_weights = element.cell.get_quadrature_weights(
                    # element.finite_element.construction_integration_order
                    element.finite_element.computation_integration_order
                )
                for qc in range(cell_quadrature_size):
                    x_q_c = cell_quadrature_points[:, qc]
                    for x_dir in range(self.field.euclidean_dimension):
                        res_qdp_file.write("{},".format(x_q_c[x_dir]))
                    for u_dir in range(self.field.field_dimension):
                        # quad_point_field_value = element.get_cell_field_increment_value(
                        #     point=x_q_c,
                        #     direction=u_dir,
                        #     field=self.field,
                        #     finite_element=self.finite_element,
                        #     element_unknown_vector=unknown_increment,
                        # )
                        quad_point_field_value = element.get_cell_field_value(
                            faces_unknown_vector=faces_unknown_vector, point=x_q_c, direction=u_dir,
                        )
                        res_qdp_file.write("{},".format(quad_point_field_value))
                    for g_dir in range(self.field.gradient_dimension):
                        strain_component = material.mat_data.s1.gradients[qp, g_dir]
                        res_qdp_file.write("{},".format(strain_component))
                    for g_dir in range(self.field.gradient_dimension):
                        if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
                            if self.field.field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN, FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS, FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC]:
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
                                # F_T_inv = np.linalg.inv(F.T)
                                sig = (1.0 / J) * PK @ F.T
                                sig_vect = np.zeros((5,), dtype=real)
                                sig_vect[0] = sig[0, 0]
                                sig_vect[1] = sig[1, 1]
                                sig_vect[2] = sig[2, 2]
                                sig_vect[3] = sig[0, 1]
                                sig_vect[4] = sig[1, 0]
                                stress_component = sig_vect[g_dir]
                            elif self.field.field_type == FieldType.DISPLACEMENT_LARGE_STRAIN:
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
                                # F_T_inv = np.linalg.inv(F.T)
                                sig = (1.0 / J) * PK @ F.T
                                sig_vect = np.zeros((9,), dtype=real)
                                sig_vect[0] = sig[0, 0]
                                sig_vect[1] = sig[1, 1]
                                sig_vect[2] = sig[2, 2]
                                sig_vect[3] = sig[0, 1]
                                sig_vect[4] = sig[1, 0]
                                sig_vect[5] = sig[0, 2]
                                sig_vect[6] = sig[2, 0]
                                sig_vect[7] = sig[1, 2]
                                sig_vect[8] = sig[2, 1]
                                stress_component = sig_vect[g_dir]
                        elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
                            stress_component = material.mat_data.s1.thermodynamic_forces[qp, g_dir]
                        res_qdp_file.write("{},".format(stress_component))
                    hyrdostatic_pressure = 0.0
                    strain_trace = 0.0
                    if self.field.field_type in [
                        FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
                        FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS,
                        FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
                        FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS,
                    ]:
                        num_diagonal_components = 3
                    else:
                        num_diagonal_components = self.field.field_dimension
                    for x_dir in range(num_diagonal_components):
                        strain_trace += material.mat_data.s1.gradients[qp, x_dir]
                        hyrdostatic_pressure += material.mat_data.s1.thermodynamic_forces[qp, x_dir]
                    hyrdostatic_pressure = hyrdostatic_pressure / num_diagonal_components
                    res_qdp_file.write("{},".format(strain_trace))
                    res_qdp_file.write("{},".format(hyrdostatic_pressure))
                    # try:
                    #     isv = material.mat_data.s1.internal_state_variables[qp]
                    #     for isv_val in range(len(isv)):
                    #         res_qdp_file.write("{},".format(isv[isv_val]))
                    # except:
                    #     pass
                    # if material.behaviour_name != "Elasticity":
                    if material.behaviour_name not in ["Elasticity", "Signorini"]:
                        isv = material.mat_data.s1.internal_state_variables[qp]
                        for isv_val in range(len(isv)):
                            res_qdp_file.write("{},".format(isv[isv_val]))
                    qp += 1
                    res_qdp_file.write("\n")
        return

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
            _constrained_system_size, _system_size = self.get_total_system_size()
            _cell_global_index_0 = _constrained_system_size + cell_index * self.finite_element.cell_basis_l.dimension * self.field.field_dimension
            _cell_global_index_1 = _constrained_system_size + (cell_index + 1) * self.finite_element.cell_basis_l.dimension * self.field.field_dimension
            _cell_range = [_cell_global_index_0, _cell_global_index_1]
            element = Element(
                _cell_range,
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
        print("ELEMNT BUILD TIME : {} | ORDERS : {}, {}".format(mean_time, self.finite_element.face_basis_k.polynomial_order, self.finite_element.cell_basis_l.polynomial_order))
        return elements

    def __check_loads(self, loads: List[Load]):
        """

        Args:
            loads:

        Returns:

        """
        if loads is None:
            return
        if isinstance(loads, list):
            if self.field.field_dimension >= len(loads) > 0:
                for i in range(len(loads)):
                    if isinstance(loads[i], Load):
                        if loads[i].direction < self.field.field_dimension:
                            continue
                        else:
                            raise ValueError
                    else:
                        raise TypeError("loads must be a list of Load")
            else:
                ValueError("loads must be a list of Load of size =< {}".format(self.field.field_dimension))
        else:
            raise TypeError("loads must be a list of Load of size =< {}".format(self.field.field_dimension))
        return

    def __check_boundary_conditions(self, boundary_conditions: List[BoundaryCondition]):
        """

        Args:
            boundary_conditions:

        Returns:

        """
        if isinstance(boundary_conditions, list):
            for boundary_condition in boundary_conditions:
                if isinstance(boundary_condition, BoundaryCondition):
                    if boundary_condition.boundary_name in self.mesh.faces_boundaries_connectivity.keys():
                        if boundary_condition.direction < self.field.field_dimension:
                            continue
                        else:
                            raise ValueError
                    else:
                        print(self.mesh.faces_boundaries_connectivity)
                        for keyy, itemm in self.mesh.faces_boundaries_connectivity.items():
                            print(keyy)
                            print(itemm)
                        for bc in boundary_conditions:
                            print(bc.boundary_name)
                        raise KeyError
                else:
                    raise TypeError
        else:
            raise TypeError
        return
