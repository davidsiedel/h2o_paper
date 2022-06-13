import numpy as np

from h2o.h2o import *
from h2o.geometry.shape import Shape
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.gradient as gradient_operator
import h2o.fem.element.operators.stabilization as stabilization_operator
import h2o.fem.element.operators.identity as identity_operator
from h2o.problem.material import Material


class DamageElement:
    cell: Shape
    faces: List[Shape]
    faces_indices: List[int]
    quad_p_indices: List[int]
    field: Field
    finite_element: FiniteElement
    element_size: int
    gradients_operators: ndarray
    stabilization_operator: ndarray
    identity_operators: ndarray
    cell_unknown_vector: ndarray

    def __init__(
        self, field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], faces_indices: List[int], quad_p_indices: List[int]
    ):
        """
        Args:
            field:
            finite_element:
            cell:
            faces:
            faces_indices:
        """
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
        self.cell_unknown_vector = np.zeros((_dx * _cl,), dtype=real)
        self.cell_unknown_vector_backup = np.zeros((_dx * _cl,), dtype=real)
        self.local_cell_unknown_vector_backup = np.zeros((_dx * _cl,), dtype=real)
        # --- BUILD OPERATORS
        self.operators = self.get_operators()
        self.stabilization_operator = stabilization_operator.get_stabilization_operator(field, finite_element, cell, faces)
        return

    def get_operators(self):
        """

        Returns:

        """
        gradients_operators = gradient_operator.get_gradient_operators(self.field, self.finite_element, self.cell, self.faces)
        identity_operators = identity_operator.get_identity_operators(self.field, self.finite_element, self.cell, self.faces)
        op_size: int = gradients_operators.shape[1] + identity_operators.shape[1]
        operators: ndarray = np.zeros((gradients_operators.shape[0], op_size, self.element_size), dtype=real)
        for i in range(gradients_operators.shape[0]):
            operators[i, : gradients_operators.shape[1], :] = gradients_operators[i]
            operators[i, gradients_operators.shape[1] :, :] = identity_operators[i]
        return operators

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
        element_unknown_vector[:_ci] += self.cell_unknown_vector
        return element_unknown_vector

    def get_transformation_gradient(self, faces_unknown_vector: ndarray, _qc: int):
        """

        Args:
            faces_unknown_vector:
            _qc:

        Returns:

        """
        element_unknown_vector = self.get_element_unknown_vector(faces_unknown_vector)
        transformation_gradient = self.operators[_qc] @ element_unknown_vector
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