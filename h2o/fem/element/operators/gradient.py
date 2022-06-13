import numpy as np

from h2o.h2o import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
import h2o.fem.element.operators.stabilization as stabilization
import sys

grad_axi_choice = 0

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=100, suppress=True,
                    threshold=sys.maxsize, formatter=None)


def get_symmetric_cartesian_gradient_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    local_grad_matric = np.zeros((_ck, _es), dtype=real)
    x_c = cell.get_centroid()
    h_c = cell.get_diameter()
    bdc = cell.get_bounding_box()
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    m_mas = np.zeros((_ck, _ck), dtype=real)
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
        d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
        d_phi_l_i = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _i)
        m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        _c0 = _j * _cl
        _c1 = (_j + 1) * _cl
        local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_c * np.tensordot(phi_k, d_phi_l_i, axes=0)
    m_mas_inv = np.linalg.inv(m_mas)
    for _f, face in enumerate(faces):
        # --- FACE GEOMETRY
        x_f = face.get_centroid()
        bdf_proj = face.get_face_bounding_box()
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
        if dist_in_face > 0:
            normal_vector_component_j = face_rotation_matrix[-1, _j]
            normal_vector_component_i = face_rotation_matrix[-1, _i]
        else:
            normal_vector_component_j = -face_rotation_matrix[-1, _j]
            normal_vector_component_i = -face_rotation_matrix[-1, _i]
        _io = finite_element.construction_integration_order
        _f_is = face.get_quadrature_size(_io)
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        for qf in range(_f_is):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, bdc)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, phi_l,
                                                                                axes=0) * normal_vector_component_j
            _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
            _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, psi_k,
                                                                                axes=0) * normal_vector_component_j
            _c0 = _j * _cl
            _c1 = (_j + 1) * _cl
            local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, phi_l,
                                                                                axes=0) * normal_vector_component_i
            _c0 = _dx * _cl + _f * _dx * _fk + _j * _fk
            _c1 = _dx * _cl + _f * _dx * _fk + (_j + 1) * _fk
            local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, psi_k,
                                                                                axes=0) * normal_vector_component_i
    local_grad_matric2 = m_mas_inv @ local_grad_matric
    return local_grad_matric2


def get_regular_cartesian_gradient_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    local_grad_matric = np.zeros((_ck, _es), dtype=real)
    x_c = cell.get_centroid()
    bdc = cell.get_bounding_box()
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    m_mas = np.zeros((_ck, _ck), dtype=real)
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
        d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
        m_mas += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        local_grad_matric[:, _c0:_c1] += w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
    m_mas_inv = np.linalg.inv(m_mas)
    for _f, face in enumerate(faces):
        # --- FACE GEOMETRY
        x_f = face.get_centroid()
        bdf_proj = face.get_face_bounding_box()
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
        if dist_in_face > 0:
            normal_vector_component_j = face_rotation_matrix[-1, _j]
        else:
            normal_vector_component_j = -face_rotation_matrix[-1, _j]
        _io = finite_element.construction_integration_order
        _f_is = face.get_quadrature_size(_io)
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        for qf in range(_f_is):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_f, x_c, bdc)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            local_grad_matric[:, _c0:_c1] -= w_q_f * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_j
            _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
            _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            local_grad_matric[:, _c0:_c1] += w_q_f * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_j
    local_grad_matric2 = m_mas_inv @ local_grad_matric
    return local_grad_matric2


def get_symmetric_axisymmetric_gradient_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _cr = finite_element.cell_basis_r.dimension
    _ck = finite_element.cell_basis_k.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    local_grad_matric = np.zeros((_ck, _es), dtype=real)
    if (_i != 2 and _j != 2):
        x_c = cell.get_centroid()
        bdc = cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        _c_is = cell.get_quadrature_size(_io)
        cell_quadrature_points = cell.get_quadrature_points(_io)
        cell_quadrature_weights = cell.get_quadrature_weights(_io)
        m_mas = np.zeros((_ck, _ck), dtype=real)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
            d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
            d_phi_l_i = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _i)
            m_mas += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(phi_k, phi_k, axes=0)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            local_grad_matric[:, _c0:_c1] += 2.0 * np.pi * (1.0 / 2.0) * x_q_c[0] * w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
            _c0 = _j * _cl
            _c1 = (_j + 1) * _cl
            local_grad_matric[:, _c0:_c1] += 2.0 * np.pi * (1.0 / 2.0) * x_q_c[0] * w_q_c * np.tensordot(phi_k, d_phi_l_i, axes=0)
        m_mas_inv = np.linalg.inv(m_mas)
        for _f, face in enumerate(faces):
            # --- FACE GEOMETRY
            x_f = face.get_centroid()
            bdf_proj = face.get_face_bounding_box()
            face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
            dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
            if dist_in_face > 0:
                normal_vector_component_j = face_rotation_matrix[-1, _j]
                normal_vector_component_i = face_rotation_matrix[-1, _i]
            else:
                normal_vector_component_j = -face_rotation_matrix[-1, _j]
                normal_vector_component_i = -face_rotation_matrix[-1, _i]
            _io = finite_element.construction_integration_order
            _f_is = face.get_quadrature_size(_io)
            face_quadrature_points = face.get_quadrature_points(_io)
            face_quadrature_weights = face.get_quadrature_weights(_io)
            for qf in range(_f_is):
                x_q_fp = np.copy(face_quadrature_points[:, qf])
                if x_q_fp[0] < 1.e-10:
                    x_q_fp[0] = 1.e-10
                w_q_f = face_quadrature_weights[qf]
                s_f = (face_rotation_matrix @ x_f)[:-1]
                s_q_f = (face_rotation_matrix @ x_q_fp)[:-1]
                coef = 2.0 * np.pi * x_q_fp[0] * w_q_f
                bdf_proj = face.get_face_bounding_box()
                phi_k = finite_element.cell_basis_k.evaluate_function(x_q_fp, x_c, bdc)
                phi_l = finite_element.cell_basis_l.evaluate_function(x_q_fp, x_c, bdc)
                psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                local_grad_matric[:, _c0:_c1] -= coef * (1.0 / 2.0) * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_j
                # local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_j
                _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                local_grad_matric[:, _c0:_c1] += coef * (1.0 / 2.0) * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_j
                # local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_j
                _c0 = _j * _cl
                _c1 = (_j + 1) * _cl
                local_grad_matric[:, _c0:_c1] -= coef * (1.0 / 2.0) * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_i
                # local_grad_matric[:, _c0:_c1] -= (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_i
                _c0 = _dx * _cl + _f * _dx * _fk + _j * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_j + 1) * _fk
                local_grad_matric[:, _c0:_c1] += coef * (1.0 / 2.0) * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_i
                # local_grad_matric[:, _c0:_c1] += (1.0 / 2.0) * w_q_f * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_i
        local_grad_matric2 = m_mas_inv @ local_grad_matric
        return local_grad_matric2
    else:
        x_c = cell.get_centroid()
        bdc = cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        _c_is = cell.get_quadrature_size(_io)
        cell_quadrature_points = cell.get_quadrature_points(_io)
        cell_quadrature_weights = cell.get_quadrature_weights(_io)
        m_mas = np.zeros((_ck, _ck), dtype=real)
        stabi = stabilization.get_axisymmetric_displacement_reconstruction_component_matrix(field, finite_element, cell,
                                                                                            faces)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
            phi_r = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
            m_mas += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(phi_k, phi_k, axes=0)
            _c0 = 0 * _cl
            _c1 = (0 + 1) * _cl
            if grad_axi_choice == 0:
                local_grad_matric[:, _c0:_c1] += (w_q_c * 2.0 * np.pi * x_q_c[0]) * (1.0 / x_q_c[0]) * np.tensordot(phi_k, phi_l, axes=0)
            elif grad_axi_choice == 1:
                _c0r = 0 * _cr
                _c1r = (0 + 1) * _cr
                local_grad_matric += (w_q_c * 2.0 * np.pi * x_q_c[0]) * (1.0 / x_q_c[0]) * np.tensordot(phi_k, phi_r, axes=0) @ stabi[_c0r:_c1r, :]
        m_mas_inv = np.linalg.inv(m_mas)
        local_grad_matric2 = m_mas_inv @ local_grad_matric
        return local_grad_matric2


def get_regular_axisymmetric_gradient_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape], _i: int, _j: int
) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _cr = finite_element.cell_basis_r.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    local_grad_matric = np.zeros((_ck, _es), dtype=real)
    if (_i != 2 and _j != 2):
        x_c = cell.get_centroid()
        bdc = cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        _c_is = cell.get_quadrature_size(_io)
        cell_quadrature_points = cell.get_quadrature_points(_io)
        cell_quadrature_weights = cell.get_quadrature_weights(_io)
        m_mas = np.zeros((_ck, _ck), dtype=real)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
            d_phi_l_j = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
            m_mas += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(phi_k, phi_k, axes=0)
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            local_grad_matric[:, _c0:_c1] += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(phi_k, d_phi_l_j, axes=0)
        m_mas_inv = np.linalg.inv(m_mas)
        for _f, face in enumerate(faces):
            # --- FACE GEOMETRY
            x_f = face.get_centroid()
            bdf_proj = face.get_face_bounding_box()
            face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
            dist_in_face = (face_rotation_matrix @ (x_f - x_c))[-1]
            if dist_in_face > 0:
                normal_vector_component_j = face_rotation_matrix[-1, _j]
            else:
                normal_vector_component_j = -face_rotation_matrix[-1, _j]
            _io = finite_element.construction_integration_order
            _f_is = face.get_quadrature_size(_io)
            face_quadrature_points = face.get_quadrature_points(_io)
            face_quadrature_weights = face.get_quadrature_weights(_io)
            for qf in range(_f_is):
                x_q_fp = np.copy(face_quadrature_points[:, qf])
                if x_q_fp[0] < 1.e-10:
                    x_q_fp[0] = 1.e-10
                w_q_f = face_quadrature_weights[qf]
                s_f = (face_rotation_matrix @ x_f)[:-1]
                s_q_f = (face_rotation_matrix @ x_q_fp)[:-1]
                coef = 2.0 * np.pi * x_q_fp[0] * w_q_f
                bdf_proj = face.get_face_bounding_box()
                phi_k = finite_element.cell_basis_k.evaluate_function(x_q_fp, x_c, bdc)
                phi_l = finite_element.cell_basis_l.evaluate_function(x_q_fp, x_c, bdc)
                psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                local_grad_matric[:, _c0:_c1] -= coef * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_j
                # local_grad_matric[:, _c0:_c1] -= w_q_f * np.tensordot(phi_k, phi_l, axes=0) * normal_vector_component_j
                _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                local_grad_matric[:, _c0:_c1] += coef * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_j
                # local_grad_matric[:, _c0:_c1] += w_q_f * np.tensordot(phi_k, psi_k, axes=0) * normal_vector_component_j
        local_grad_matric2 = m_mas_inv @ local_grad_matric
        return local_grad_matric2
    else:
        x_c = cell.get_centroid()
        bdc = cell.get_bounding_box()
        _io = finite_element.construction_integration_order
        _c_is = cell.get_quadrature_size(_io)
        cell_quadrature_points = cell.get_quadrature_points(_io)
        cell_quadrature_weights = cell.get_quadrature_weights(_io)
        m_mas = np.zeros((_ck, _ck), dtype=real)
        stabi = stabilization.get_axisymmetric_displacement_reconstruction_component_matrix(field, finite_element, cell,
                                                                                            faces)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_k = finite_element.cell_basis_k.evaluate_function(x_q_c, x_c, bdc)
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
            phi_r = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
            m_mas += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(phi_k, phi_k, axes=0)
            _c0 = 0 * _cl
            _c1 = (0 + 1) * _cl
            if grad_axi_choice == 0:
                local_grad_matric[:, _c0:_c1] += 2.0 * np.pi * x_q_c[0] * w_q_c * (1.0 / x_q_c[0]) * np.tensordot(phi_k, phi_l, axes=0)
            elif grad_axi_choice == 1:
                _c0r = 0 * _cr
                _c1r = (0 + 1) * _cr
                local_grad_matric += 2.0 * np.pi * x_q_c[0] * w_q_c * (1.0 / x_q_c[0]) * np.tensordot(phi_k, phi_r, axes=0) @ stabi[_c0r:_c1r, :]
        m_mas_inv = np.linalg.inv(m_mas)
        local_grad_matric2 = m_mas_inv @ local_grad_matric
        return local_grad_matric2


def get_gradient_operators(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _ck = finite_element.cell_basis_k.dimension
    _cr = finite_element.cell_basis_r.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    _gs = field.gradient_dimension
    # --- INTEGRATION ORDER
    _io = finite_element.computation_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    gradient_operators = np.zeros((_c_is, _gs, _es), dtype=real)
    # --- CELL GEOMETRY
    bdc = cell.get_bounding_box()
    x_c = cell.get_centroid()
    for _qc in range(_c_is):
        x_qc = cell_quadrature_points[:, _qc]
        v_ck = finite_element.cell_basis_k.evaluate_function(x_qc, x_c, bdc)
        v_r = finite_element.cell_basis_r.evaluate_function(x_qc, x_c, bdc)
        for key, val in field.voigt_data.items():
            _i = key[0]
            _j = key[1]
            voigt_indx = val[0]
            voigt_coef = val[1]
            if field.derivation_type == DerivationType.REGULAR:
                gradient_component_matrix = get_regular_cartesian_gradient_component_matrix(
                    field=field, finite_element=finite_element, cell=cell, faces=faces, _i=_i, _j=_j
                )
            elif field.derivation_type == DerivationType.SYMMETRIC:
                gradient_component_matrix = get_symmetric_cartesian_gradient_component_matrix(
                    field=field, finite_element=finite_element, cell=cell, faces=faces, _i=_i, _j=_j
                )
            elif field.derivation_type == DerivationType.LARGE_STRAIN_AXISYMMETRIC:
                gradient_component_matrix = get_regular_axisymmetric_gradient_component_matrix(
                    field=field, finite_element=finite_element, cell=cell, faces=faces, _i=_i, _j=_j
                )
            elif field.derivation_type == DerivationType.SMALL_STRAIN_AXISYMMETRIC:
                gradient_component_matrix = get_symmetric_axisymmetric_gradient_component_matrix(
                    field=field, finite_element=finite_element, cell=cell, faces=faces, _i=_i, _j=_j
                )
            else:
                raise KeyError
            if (_i != 2 and _j != 2):
                gradient_operators[_qc, voigt_indx] = voigt_coef * v_ck @ gradient_component_matrix
            else:
                if (grad_axi_choice == 2):
                    # print("THERE : {} {}".format(_i,_j))
                    stabi = stabilization.get_axisymmetric_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
                    gradient_operators[_qc, voigt_indx] = voigt_coef * (1.0 / x_qc[0]) * v_r @ stabi[:_cr, :]
                else:
                    gradient_operators[_qc, voigt_indx] = voigt_coef * v_ck @ gradient_component_matrix
    return gradient_operators
