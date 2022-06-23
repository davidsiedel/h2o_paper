import numpy as np

import tfel
import tfel.math

from h2o.h2o import *
from h2o.geometry.shape import Shape
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.gradient as gradient_operator
import h2o.fem.element.operators.stabilization as stabilization_operator
import h2o.fem.element.operators.identity as identity_operator
from h2o.problem.material import Material


class Element:
    cell: Shape
    faces: List[Shape]
    faces_indices: List[int]
    quad_p_indices: List[int]
    field: Field
    finite_element: FiniteElement
    element_size: int
    gradients_operators: ndarray
    stabilization_operator: ndarray

    def __init__(
        self, cell_range: List[int], field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], faces_indices: List[int], quad_p_indices: List[int]
    ):
        """
        Args:
            field:
            finite_element:
            cell:
            faces:
            faces_indices:
        """
        self.cell_range = cell_range
        self.cell = cell
        self.faces = faces
        self.faces_indices = faces_indices
        self.quad_p_indices = quad_p_indices
        self.field = field
        self.finite_element = finite_element
        # --- GET INDICES
        _dx = field.field_dimension
        _cl = finite_element.cell_basis_l.dimension
        _fk = finite_element.face_basis_k.dimension
        _nf = len(self.faces)
        # --- BUILD LOCAL MATRICES AND VECTORS
        self.element_size = _dx * (_cl + _nf * _fk)
        self.m_cell_cell_inv = np.zeros((_cl * _dx, _cl * _dx), dtype=real)
        self.m_cell_faces = np.zeros((_cl * _dx, _nf * _fk * _dx), dtype=real)
        self.v_cell = np.zeros((_cl * _dx,), dtype=real)
        # --- BUILD OPERATORS
        self.gradients_operators = gradient_operator.get_gradient_operators(field, finite_element, cell, faces)
        self.stabilization_operator = stabilization_operator.get_stabilization_operator(field, finite_element, cell, faces)
        self.identity_operators = identity_operator.get_identity_operators(field, finite_element, cell, faces)
        self.operators = self.get_operators()
        self.accelerator = tfel.math.UAnderson(3, 1)
        return

    def get_operators(self):
        if self.field.field_type == FieldType.SCALAR_PLANE:
            op_size: int = self.gradients_operators.shape[1] + self.identity_operators.shape[1]
            operators: ndarray = np.zeros((self.gradients_operators.shape[0], op_size, self.element_size), dtype=real)
            for i in range(self.gradients_operators.shape[0]):
                operators[i, : self.gradients_operators.shape[1], :] = self.gradients_operators[i]
                operators[i, self.gradients_operators.shape[1] :, :] = self.identity_operators[i]
            # print(operators)
            return operators
        else:
            operators = self.gradients_operators
            return operators

    def get_stress_field(self, problem, material: Material):
        _io: int = problem.finite_element.computation_integration_order
        cell_quadrature_size = self.cell.get_quadrature_size(
            _io, quadrature_type=problem.quadrature_type
        )
        cell_quadrature_points = self.cell.get_quadrature_points(
            _io, quadrature_type=problem.quadrature_type
        )
        cell_quadrature_weights = self.cell.get_quadrature_weights(
            _io, quadrature_type=problem.quadrature_type
        )
        _ck = problem.finite_element.cell_basis_k.dimension
        m_mas: ndarray = np.zeros((_ck, _ck), dtype=real)
        for _qc in range(cell_quadrature_size):
            _qp = self.quad_p_indices[_qc]
            _w_q_c = cell_quadrature_weights[_qc]
            _x_q_c = cell_quadrature_points[:, _qc]

    def solve_active_set(self, field: Field, finite_element: FiniteElement, tangent_mat: ndarray):
        _cl = finite_element.cell_basis_l.dimension
        active_set_size: int = 0
        inactive_set_size: int = _cl
        lagrange_mat: ndarray = np.eye(_cl)
        total_mat: ndarray = np.zeros((2 * _cl, 2 * _cl), dtype=real)
        total_mat[:_cl, :_cl] = tangent_mat
        total_mat[_cl:, :_cl] = lagrange_mat
        total_mat[:_cl, _cl:] = lagrange_mat
        _dx = field.field_dimension
        B_mat: ndarray = np.zeros((self.element_size, self.element_size), dtype=real)
        x_c = self.cell.get_centroid()
        bdc = self.cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        _c_is = self.cell.get_quadrature_size(_io)
        cell_quadrature_points = self.cell.get_quadrature_points(_io)
        cell_quadrature_weights = self.cell.get_quadrature_weights(_io)
        for _qc in range(_c_is):
            w_q_c = cell_quadrature_weights[_qc]
            B_mat += w_q_c * self.identity_operators[_qc].T @ np.eye(_dx) @ self.identity_operators[_qc]
        return

    def get_element_unknown_vector(self, faces_global_unknown_vector: ndarray) -> ndarray:
        """

        Args:
            faces_global_unknown_vector:

        Returns:

        """
        _dx = self.field.field_dimension
        _fk = self.finite_element.face_basis_k.dimension
        _cl = self.finite_element.cell_basis_l.dimension
        element_unknown_vector = np.zeros((self.element_size,), dtype=real)
        for _f_local, _f_global in enumerate(self.faces_indices):
            _ci_g = _f_global * (_fk * _dx)
            _cj_g = (_f_global + 1) * (_fk * _dx)
            _ci_l = (_dx * _cl) + _f_local * (_fk * _dx)
            _cj_l = (_dx * _cl) + (_f_local + 1) * (_fk * _dx)
            element_unknown_vector[_ci_l:_cj_l] += faces_global_unknown_vector[_ci_g:_cj_g]
        _ci = _cl * _dx
        element_unknown_vector[:_ci] += faces_global_unknown_vector[self.cell_range[0]:self.cell_range[1]]
        return element_unknown_vector


    def get_transformation_gradient(self, faces_unknown_vector: ndarray, _qc: int):
        """

        Args:
            faces_unknown_vector:
            _qc:

        Returns:

        """
        element_unknown_vector = self.get_element_unknown_vector(faces_unknown_vector)
        if self.field.grad_type == GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT:
            ones_vect = np.zeros((self.field.gradient_dimension,), dtype=real)
            for indx in range(3):
                ones_vect[indx] = 1.0
            # transformation_gradient_0 = self.gradients_operators[_qc] @ element_unknown_vector
            transformation_gradient_0 = self.operators[_qc] @ element_unknown_vector
            transformation_gradient = ones_vect + transformation_gradient_0
        elif self.field.grad_type == GradType.DISPLACEMENT_SMALL_STRAIN:
            # transformation_gradient_0 = self.gradients_operators[_qc] @ element_unknown_vector
            transformation_gradient_0 = self.operators[_qc] @ element_unknown_vector
            transformation_gradient = transformation_gradient_0
        else:
            raise KeyError("No such strain type")
        return transformation_gradient

    def get_field_gradient(self, faces_unknown_vector: ndarray, _qc: int):
        """

        Args:
            faces_unknown_vector:
            _qc:

        Returns:

        """
        element_unknown_vector = self.get_element_unknown_vector(faces_unknown_vector)
        transformation_gradient = self.gradients_operators[_qc] @ element_unknown_vector
        return transformation_gradient

    def get_cell_field_value(self, faces_unknown_vector: ndarray, point: ndarray, direction: int) -> float:
        """

        Args:
            faces_unknown_vector:
            point:
            direction:

        Returns:

        """
        element_unknown_vector = self.get_element_unknown_vector(faces_unknown_vector)
        _cl = self.finite_element.cell_basis_l.dimension
        _c0 = direction * _cl
        _c1 = (direction + 1) * _cl
        x_c = self.cell.get_centroid()
        h_c = self.cell.get_diameter()
        bdc = self.cell.get_bounding_box()
        # vcl = self.finite_element.cell_basis_l.evaluate_function(point, x_c, h_c)
        vcl = self.finite_element.cell_basis_l.evaluate_function(point, x_c, bdc)
        field_unknown_value = vcl @ element_unknown_vector[_c0:_c1]
        return field_unknown_value

    def get_diskpp_notation(self, field: Field, finite_element: FiniteElement, index: int) -> int:
        """

        Args:
            field:
            finite_element:
            index:

        Returns:

        """
        _dx = field.field_dimension
        _cl = finite_element.cell_basis_l.dimension
        _fk = finite_element.face_basis_k.dimension
        _nf = len(self.faces)
        for _f in range(_nf):
            for i in range(_dx):
                l0: int = i * _cl
                l1: int = (i + 1) * _cl
                if index in range(l0, l1):
                    l_h2o: int = index - l0
                    l_disk: int = i + l_h2o * _dx
                    return l_disk
                l0: int = _dx * _cl + _f * _fk * _dx + i * _fk
                l1: int = _dx * _cl + _f * _fk * _dx + (i + 1) * _fk
                if index in range(l0, l1):
                    l_h2o: int = index - l0
                    l_disk: int = i + l_h2o * _dx
                    if _f == 0:
                        _f_disk: int = 1
                    if _f == 1:
                        _f_disk: int = 3
                    if _f == 2:
                        _f_disk: int = 2
                    if _f == 3:
                        _f_disk: int = 0
                    # return _dx * _cl + _f * _fk * _dx + l_disk
                    return _dx * _cl + _f_disk * _fk * _dx + l_disk