import numpy as np
from numpy import ndarray

from h2o.h2o import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
# from h2o.fem.element.operators.operator_gradient import get_regular_gradient_component_matrix, get_symmetric_gradient_component_matrix
import sys

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=8, suppress=True,
                    threshold=sys.maxsize, formatter=None)

def get_cartesian_displacement_reconstruction_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]
) -> ndarray:
    # --- ELEMENT DATA
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _cr = finite_element.cell_basis_r.dimension
    _cr_star = finite_element.cell_basis_r.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    # --- CELL GEOMETRY
    x_c = cell.get_centroid()
    bdc = cell.get_bounding_box()
    # --- INITIATE
    local_recons_matric2: ndarray = np.zeros((_dx * _cr, _es), dtype=real)
    reconstruction_op = np.zeros((_dx * _cr, _es), dtype=real)
    # --- CELL ENVIRONMENT
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    m_stif: ndarray = np.zeros((_cr, _cr), dtype=real)
    for _i in range(_dx):
        local_reconstruction_matric = np.zeros((_cr, _es), dtype=real)
        for _j in range(_d):
            for qc in range(_c_is):
                x_q_c: ndarray = cell_quadrature_points[:, qc]
                w_q_c: float = cell_quadrature_weights[qc]
                d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
                d_phi_l_j: ndarray = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
                m_stif += w_q_c * np.tensordot(d_phi_r_j, d_phi_r_j, axes=0)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                local_reconstruction_matric[:, _c0:_c1] += w_q_c * np.tensordot(d_phi_r_j, d_phi_l_j, axes=0)
        m_stif_inv: ndarray = np.linalg.inv(m_stif[1:, 1:])
        for _j in range(_d):
            for _f, face in enumerate(faces):
                x_f = face.get_centroid()
                bdf_proj = face.get_face_bounding_box()
                face_rotation_matrix = face.get_rotation_matrix().copy()
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
                    d_phi_r_j = finite_element.cell_basis_r.evaluate_derivative(x_q_f, x_c, bdc, _j)
                    phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
                    psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                    _c0 = _i * _cl
                    _c1 = (_i + 1) * _cl
                    local_reconstruction_matric[:, _c0:_c1] -= w_q_f * np.tensordot(d_phi_r_j, phi_l, axes=0) * normal_vector_component_j
                    _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                    _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                    local_reconstruction_matric[:, _c0:_c1] += w_q_f * np.tensordot(d_phi_r_j, psi_k, axes=0) * normal_vector_component_j
        _r0 = _i * _cr
        _r1 = (_i + 1) * _cr
        local_recons_matric2[_r0 + 1:_r1, :] = m_stif_inv @ local_reconstruction_matric[1:, :]
        # --- CONSTANT FIX
        interpolator_component_matrix = np.zeros((_cr, _es), dtype=real)
        m_fill = np.eye(_cl, dtype=real)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        interpolator_component_matrix[0:_cl, _c0:_c1] = m_fill
        linop = np.zeros((_es,), dtype=real)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_r = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
            linop += (1 / _c_is) * w_q_c * phi_r @ (interpolator_component_matrix - local_recons_matric2[_r0:_r1, :])
        reconstruction_op[_r0, :] = linop
        reconstruction_op[_r0 + 1:_r1, :] = local_recons_matric2[_r0 + 1:_r1, :]
    return reconstruction_op

def get_axisymmetric_displacement_reconstruction_component_matrix(
        field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]
) -> ndarray:
    # --- ELEMENT DATA
    _d = field.euclidean_dimension
    _dx = field.field_dimension
    _cl = finite_element.cell_basis_l.dimension
    _cr = finite_element.cell_basis_r.dimension
    _cr_star = finite_element.cell_basis_r.dimension
    _fk = finite_element.face_basis_k.dimension
    _nf = len(faces)
    _es = _dx * (_cl + _nf * _fk)
    # --- CELL GEOMETRY
    x_c = cell.get_centroid()
    bdc = cell.get_bounding_box()
    # --- INITIATE
    local_recons_matric2: ndarray = np.zeros((_dx * _cr, _es), dtype=real)
    reconstruction_op = np.zeros((_dx * _cr, _es), dtype=real)
    # --- CELL ENVIRONMENT
    _io = finite_element.construction_integration_order
    _c_is = cell.get_quadrature_size(_io)
    cell_quadrature_points = cell.get_quadrature_points(_io)
    cell_quadrature_weights = cell.get_quadrature_weights(_io)
    m_stif: ndarray = np.zeros((_cr, _cr), dtype=real)
    for _i in range(_dx):
        local_reconstruction_matric = np.zeros((_cr, _es), dtype=real)
        for _j in range(_d):
            for qc in range(_c_is):
                x_q_c: ndarray = cell_quadrature_points[:, qc]
                w_q_c: float = cell_quadrature_weights[qc]
                d_phi_r_j: ndarray = finite_element.cell_basis_r.evaluate_derivative(x_q_c, x_c, bdc, _j)
                d_phi_l_j: ndarray = finite_element.cell_basis_l.evaluate_derivative(x_q_c, x_c, bdc, _j)
                m_stif += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(d_phi_r_j, d_phi_r_j, axes=0)
                _c0 = _i * _cl
                _c1 = (_i + 1) * _cl
                local_reconstruction_matric[:, _c0:_c1] += 2.0 * np.pi * x_q_c[0] * w_q_c * np.tensordot(d_phi_r_j, d_phi_l_j, axes=0)
                if _i == 0:
                    phi_r: ndarray = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
                    phi_l: ndarray = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
                    m_stif += 2.0 * np.pi * x_q_c[0] * w_q_c * (1.0 / x_q_c[0]) * np.tensordot(phi_r, phi_r, axes=0)
                    local_reconstruction_matric[:, _c0:_c1] += 2.0 * np.pi * x_q_c[0] * w_q_c * (1.0 / x_q_c[0]) * np.tensordot(phi_r, phi_l, axes=0)
        m_stif_inv: ndarray = np.linalg.inv(m_stif[1:, 1:])
        for _j in range(_d):
            for _f, face in enumerate(faces):
                x_f = face.get_centroid()
                bdf_proj = face.get_face_bounding_box()
                face_rotation_matrix = face.get_rotation_matrix().copy()
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
                    d_phi_r_j = finite_element.cell_basis_r.evaluate_derivative(x_q_fp, x_c, bdc, _j)
                    phi_l = finite_element.cell_basis_l.evaluate_function(x_q_fp, x_c, bdc)
                    psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
                    _c0 = _i * _cl
                    _c1 = (_i + 1) * _cl
                    local_reconstruction_matric[:, _c0:_c1] -= coef * np.tensordot(d_phi_r_j, phi_l, axes=0) * normal_vector_component_j
                    _c0 = _dx * _cl + _f * _dx * _fk + _i * _fk
                    _c1 = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
                    local_reconstruction_matric[:, _c0:_c1] += coef * np.tensordot(d_phi_r_j, psi_k, axes=0) * normal_vector_component_j
        _r0 = _i * _cr
        _r1 = (_i + 1) * _cr
        local_recons_matric2[_r0 + 1:_r1, :] = m_stif_inv @ local_reconstruction_matric[1:, :]
        # --- CONSTANT FIX
        interpolator_component_matrix = np.zeros((_cr, _es), dtype=real)
        m_fill = np.eye(_cl, dtype=real)
        _c0 = _i * _cl
        _c1 = (_i + 1) * _cl
        interpolator_component_matrix[0:_cl, _c0:_c1] = m_fill
        linop = np.zeros((_es,), dtype=real)
        for qc in range(_c_is):
            x_q_c = cell_quadrature_points[:, qc]
            w_q_c = cell_quadrature_weights[qc]
            phi_r = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
            linop += 2.0 * np.pi * x_q_c[0] * w_q_c * (1 / _c_is) * phi_r @ (interpolator_component_matrix - local_recons_matric2[_r0:_r1, :])
        reconstruction_op[_r0, :] = linop
        reconstruction_op[_r0 + 1:_r1, :] = local_recons_matric2[_r0 + 1:_r1, :]
    return reconstruction_op

def get_stabilization_operator3hho(field: Field, finite_element: FiniteElement, cell: Shape,
                                   faces: List[Shape]) -> ndarray:
    # --- ELEMENT DATA
    _dx: int = field.field_dimension
    _cl: int = finite_element.cell_basis_l.dimension
    _cr: int = finite_element.cell_basis_r.dimension
    _fk: int = finite_element.face_basis_k.dimension
    _nf: int = len(faces)
    _es: int = _dx * (_cl + _nf * _fk)
    # --- INIT
    stabilization_operator: ndarray = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk * _dx, _es), dtype=real)
    # --- CELL GEOMETRY
    x_c: ndarray = cell.get_centroid()
    h_c: float = cell.get_diameter()
    bdc: ndarray = cell.get_bounding_box()
    # --- INTEGRATION
    if field.derivation_type in [DerivationType.LARGE_STRAIN_AXISYMMETRIC, DerivationType.SMALL_STRAIN_AXISYMMETRIC]:
        recop: ndarray = get_axisymmetric_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
        # recop: ndarray = get_cartesian_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
    else:
        recop: ndarray = get_cartesian_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
    # _io: int = finite_element.construction_integration_order
    # _io: int = finite_element.r_order + finite_element.l_order
    _io = finite_element.construction_integration_order
    # _io: int = 8
    _c_is: int = cell.get_quadrature_size(_io)
    cell_quadrature_points: ndarray = cell.get_quadrature_points(_io)
    cell_quadrature_weights: ndarray = cell.get_quadrature_weights(_io)
    prj_r2l_t2t_lhs: ndarray = np.zeros((_cl, _cr))
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_r = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
        phi_l = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
        prj_r2l_t2t_lhs += w_q_c * np.tensordot(phi_l, phi_r, axes=0)
    # _io: int = finite_element.l_order + finite_element.l_order
    _io = finite_element.construction_integration_order
    # _io: int = 8
    _c_is: int = cell.get_quadrature_size(_io)
    cell_quadrature_points: ndarray = cell.get_quadrature_points(_io)
    cell_quadrature_weights: ndarray = cell.get_quadrature_weights(_io)
    prj_r2l_t2t_rhs: ndarray = np.zeros((_cl, _cl))
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_l = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
        prj_r2l_t2t_rhs += w_q_c * np.tensordot(phi_l, phi_l, axes=0)
    prj_T_rhs_inv: ndarray = np.linalg.inv(prj_r2l_t2t_rhs)
    prj_r2l_t2t: ndarray = prj_T_rhs_inv @ prj_r2l_t2t_lhs
    for _f, face in enumerate(faces):
        # --- FACE GEOMETRY
        x_f = face.get_centroid()
        h_f = face.get_diameter()
        bdf = face.get_bounding_box()
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        _io = finite_element.construction_integration_order
        # _io: int = 8
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        prj_l2k_t2f_lhs: ndarray = np.zeros((_fk, _fk))
        prj_r2k_t2f_lhs: ndarray = np.zeros((_fk, _fk))
        m_mas_f: ndarray = np.zeros((_fk, _fk))
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            # bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            prj_r2k_t2f_lhs += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
            prj_l2k_t2f_lhs += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
            m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
        # _io: int = finite_element.k_order + finite_element.l_order
        _io = finite_element.construction_integration_order
        # _io: int = 8
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        prj_l2k_t2f_rhs: ndarray = np.zeros((_fk, _cl))
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            # bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_f, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            prj_l2k_t2f_rhs += w_q_f * np.tensordot(psi_k, phi_l, axes=0)
        # _io: int = finite_element.k_order + finite_element.r_order
        _io = finite_element.construction_integration_order
        # _io: int = 8
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        prj_r2k_t2f_rhs: ndarray = np.zeros((_fk, _cr))
        for qf in range(face_quadrature_size):
            x_q_f = face_quadrature_points[:, qf]
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            x_q_f_p = face_rotation_matrix @ x_q_f
            x_c_p = face_rotation_matrix @ x_c
            phi_r = finite_element.cell_basis_r.evaluate_function(x_q_f, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            prj_r2k_t2f_rhs += w_q_f * np.tensordot(psi_k, phi_r, axes=0)
        prj_r2k_t2f_lhs_inv: ndarray = np.linalg.inv(prj_r2k_t2f_lhs)
        prj_r2k_t2f: ndarray = prj_r2k_t2f_lhs_inv @ prj_r2k_t2f_rhs
        prj_l2k_t2f_lhs_inv: ndarray = np.linalg.inv(prj_l2k_t2f_lhs)
        prj_l2k_t2f: ndarray = prj_l2k_t2f_lhs_inv @ prj_l2k_t2f_rhs
        # m_mas_f_inv = np.linalg.inv(m_mas_f)
        # proj_mat = m_mas_f_inv @ m_hyb_f
        m = np.eye(_fk, dtype=real)
        for _i in range(_dx):
            interop: ndarray = np.zeros((_cl, _es), dtype=real)
            stabilization_op33 = np.zeros((_fk, _es), dtype=real)
            _ri = _i * _fk
            _rj = (_i + 1) * _fk
            _ci = _i * _cl
            _cj = (_i + 1) * _cl
            _cri = _i * _cr
            _crj = (_i + 1) * _cr
            interop[:, _ci:_cj] += np.eye(_cl, dtype=real)
            # stabilization_op33[:, _ci:_cj] -= proj_mat
            that_00: ndarray = prj_r2k_t2f @ recop[_cri: _crj, :]
            that_01: ndarray = prj_l2k_t2f @ (- prj_r2l_t2t @ recop[_cri: _crj, :] + interop)
            that: ndarray = that_00 + that_01
            stabilization_op33 -= that
            _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
            _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            stabilization_op33[:, _ci:_cj] += m
            stabilization_operator += (1.0 / h_f) * stabilization_op33.T @ m_mas_f @ stabilization_op33
    return stabilization_operator

def get_stabilization_operator3hho_axisymmetric(field: Field, finite_element: FiniteElement, cell: Shape,
                                   faces: List[Shape]) -> ndarray:
    # --- ELEMENT DATA
    _dx: int = field.field_dimension
    _cl: int = finite_element.cell_basis_l.dimension
    _cr: int = finite_element.cell_basis_r.dimension
    _fk: int = finite_element.face_basis_k.dimension
    _nf: int = len(faces)
    _es: int = _dx * (_cl + _nf * _fk)
    # --- INIT
    stabilization_operator: ndarray = np.zeros((_es, _es), dtype=real)
    stabilization_op = np.zeros((_fk * _dx, _es), dtype=real)
    # --- CELL GEOMETRY
    x_c: ndarray = cell.get_centroid()
    h_c: float = cell.get_diameter()
    bdc: ndarray = cell.get_bounding_box()
    # --- INTEGRATION
    recop: ndarray = get_axisymmetric_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
    # if field.derivation_type in [DerivationType.LARGE_STRAIN_AXISYMMETRIC, DerivationType.SMALL_STRAIN_AXISYMMETRIC]:
    #     recop: ndarray = get_axisymmetric_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
    #     # recop: ndarray = get_cartesian_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
    # else:
    #     recop: ndarray = get_cartesian_displacement_reconstruction_component_matrix(field, finite_element, cell, faces)
    # _io: int = finite_element.construction_integration_order
    # _io: int = finite_element.r_order + finite_element.l_order
    _io = finite_element.construction_integration_order
    # _io: int = 8
    _c_is: int = cell.get_quadrature_size(_io)
    cell_quadrature_points: ndarray = cell.get_quadrature_points(_io)
    cell_quadrature_weights: ndarray = cell.get_quadrature_weights(_io)
    prj_r2l_t2t_lhs: ndarray = np.zeros((_cl, _cr))
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_r = finite_element.cell_basis_r.evaluate_function(x_q_c, x_c, bdc)
        phi_l = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
        prj_r2l_t2t_lhs += 2.0 * np.pi * w_q_c * x_q_c[0] * np.tensordot(phi_l, phi_r, axes=0)
    # _io: int = finite_element.l_order + finite_element.l_order
    _io = finite_element.construction_integration_order
    # _io: int = 8
    _c_is: int = cell.get_quadrature_size(_io)
    cell_quadrature_points: ndarray = cell.get_quadrature_points(_io)
    cell_quadrature_weights: ndarray = cell.get_quadrature_weights(_io)
    prj_r2l_t2t_rhs: ndarray = np.zeros((_cl, _cl))
    for qc in range(_c_is):
        x_q_c = cell_quadrature_points[:, qc]
        w_q_c = cell_quadrature_weights[qc]
        phi_l = finite_element.cell_basis_l.evaluate_function(x_q_c, x_c, bdc)
        prj_r2l_t2t_rhs += 2.0 * np.pi * w_q_c * x_q_c[0] * np.tensordot(phi_l, phi_l, axes=0)
    prj_T_rhs_inv: ndarray = np.linalg.inv(prj_r2l_t2t_rhs)
    prj_r2l_t2t: ndarray = prj_T_rhs_inv @ prj_r2l_t2t_lhs
    for _f, face in enumerate(faces):
        # --- FACE GEOMETRY
        x_f = face.get_centroid()
        h_f = face.get_diameter()
        bdf = face.get_bounding_box()
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        _io = finite_element.construction_integration_order
        # _io: int = 8
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        prj_l2k_t2f_lhs: ndarray = np.zeros((_fk, _fk))
        prj_r2k_t2f_lhs: ndarray = np.zeros((_fk, _fk))
        prj_l2k_t2f_rhs: ndarray = np.zeros((_fk, _cl))
        prj_r2k_t2f_rhs: ndarray = np.zeros((_fk, _cr))
        m_mas_f: ndarray = np.zeros((_fk, _fk))
        for qf in range(face_quadrature_size):
            x_q_fp = np.copy(face_quadrature_points[:, qf])
            if x_q_fp[0] < 1.e-10:
                x_q_fp[0] = 1.e-10
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_fp)[:-1]
            coef = 2.0 * np.pi * x_q_fp[0] * w_q_f
            bdf_proj = face.get_face_bounding_box()
            # x_q_f = face_quadrature_points[:, qf]
            # w_q_f = face_quadrature_weights[qf]
            # s_f = (face_rotation_matrix @ x_f)[:-1]
            # s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            # coef: real = 0.0
            # if x_q_f[0] > 1.e-10:
            # # if x_q_f[0] > 1.e-8:
            #     coef = 2.0 * np.pi * x_q_f[0] * w_q_f
            # else:
            #     coef = w_q_f
            #     coef = 2.0 * np.pi * 1.e-10 * w_q_f
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_fp, x_c, bdc)
            phi_r = finite_element.cell_basis_r.evaluate_function(x_q_fp, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            prj_r2k_t2f_lhs += coef * np.tensordot(psi_k, psi_k, axes=0)
            prj_l2k_t2f_lhs += coef * np.tensordot(psi_k, psi_k, axes=0)
            m_mas_f += coef * np.tensordot(psi_k, psi_k, axes=0)
            prj_l2k_t2f_rhs += coef * np.tensordot(psi_k, phi_l, axes=0)
            prj_r2k_t2f_rhs += coef * np.tensordot(psi_k, phi_r, axes=0)
        prj_r2k_t2f_lhs_inv: ndarray = np.linalg.inv(prj_r2k_t2f_lhs)
        prj_r2k_t2f: ndarray = prj_r2k_t2f_lhs_inv @ prj_r2k_t2f_rhs
        prj_l2k_t2f_lhs_inv: ndarray = np.linalg.inv(prj_l2k_t2f_lhs)
        prj_l2k_t2f: ndarray = prj_l2k_t2f_lhs_inv @ prj_l2k_t2f_rhs
        # m_mas_f_inv = np.linalg.inv(m_mas_f)
        # proj_mat = m_mas_f_inv @ m_hyb_f
        m = np.eye(_fk, dtype=real)
        for _i in range(_dx):
            interop: ndarray = np.zeros((_cl, _es), dtype=real)
            stabilization_op33 = np.zeros((_fk, _es), dtype=real)
            _ri = _i * _fk
            _rj = (_i + 1) * _fk
            _ci = _i * _cl
            _cj = (_i + 1) * _cl
            _cri = _i * _cr
            _crj = (_i + 1) * _cr
            interop[:, _ci:_cj] += np.eye(_cl, dtype=real)
            # stabilization_op33[:, _ci:_cj] -= proj_mat
            that_00: ndarray = prj_r2k_t2f @ recop[_cri: _crj, :]
            that_01: ndarray = prj_l2k_t2f @ (- prj_r2l_t2t @ recop[_cri: _crj, :] + interop)
            that: ndarray = that_00 + that_01
            stabilization_op33 -= that
            _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
            _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            stabilization_op33[:, _ci:_cj] += m
            stabilization_operator += (1.0 / h_f) * stabilization_op33.T @ m_mas_f @ stabilization_op33
    return stabilization_operator

def get_hdg_stabilization_axisymmetric(field: Field, finite_element: FiniteElement, cell: Shape,
                                   faces: List[Shape]) -> ndarray:
    # --- ELEMENT DATA
    _dx: int = field.field_dimension
    _cl: int = finite_element.cell_basis_l.dimension
    _cr: int = finite_element.cell_basis_r.dimension
    _fk: int = finite_element.face_basis_k.dimension
    _nf: int = len(faces)
    _es: int = _dx * (_cl + _nf * _fk)
    # --- INIT
    stabilization_operator: ndarray = np.zeros((_es, _es), dtype=real)
    # --- CELL GEOMETRY
    x_c: ndarray = cell.get_centroid()
    h_c: float = cell.get_diameter()
    bdc: ndarray = cell.get_bounding_box()
    # --- INTEGRATION
    _io = finite_element.construction_integration_order
    for _f, face in enumerate(faces):
        # --- FACE GEOMETRY
        x_f = face.get_centroid()
        h_f = face.get_diameter()
        bdf = face.get_bounding_box()
        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
        _io = finite_element.construction_integration_order
        # _io: int = 8
        face_quadrature_points = face.get_quadrature_points(_io)
        face_quadrature_weights = face.get_quadrature_weights(_io)
        face_quadrature_size = face.get_quadrature_size(_io)
        prj_l2k_t2f_lhs: ndarray = np.zeros((_fk, _fk))
        prj_l2k_t2f_rhs: ndarray = np.zeros((_fk, _cl))
        m_mas_f: ndarray = np.zeros((_fk, _fk))
        for qf in range(face_quadrature_size):
            x_q_fp = np.copy(face_quadrature_points[:, qf])
            if x_q_fp[0] < 1.e-10:
                x_q_fp[0] = 1.e-10
            w_q_f = face_quadrature_weights[qf]
            s_f = (face_rotation_matrix @ x_f)[:-1]
            s_q_f = (face_rotation_matrix @ x_q_fp)[:-1]
            coef = 2.0 * np.pi * x_q_fp[0] * w_q_f
            bdf_proj = face.get_face_bounding_box()
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            # if x_q_f[0] > 0.1e-8:
            #     prj_r2k_t2f_lhs += w_q_f * x_q_f[0] * np.tensordot(psi_k, psi_k, axes=0)
            #     prj_l2k_t2f_lhs += w_q_f * x_q_f[0] * np.tensordot(psi_k, psi_k, axes=0)
            # else:
            prj_l2k_t2f_lhs += coef * np.tensordot(psi_k, psi_k, axes=0)
            # m_mas_f += w_q_f * x_q_f[0] * np.tensordot(psi_k, psi_k, axes=0)
            m_mas_f += coef * np.tensordot(psi_k, psi_k, axes=0)
            #
            # x_q_f = face_quadrature_points[:, qf]
            # w_q_f = face_quadrature_weights[qf]
            # s_f = (face_rotation_matrix @ x_f)[:-1]
            # s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
            # bdf_proj = (face_rotation_matrix @ bdf)[:-1]
            bdf_proj = face.get_face_bounding_box()
            phi_l = finite_element.cell_basis_l.evaluate_function(x_q_fp, x_c, bdc)
            psi_k = finite_element.face_basis_k.evaluate_function(s_q_f, s_f, bdf_proj)
            # prj_l2k_t2f_rhs += w_q_f * x_q_f[0] * np.tensordot(psi_k, phi_l, axes=0)
            prj_l2k_t2f_rhs += coef * np.tensordot(psi_k, phi_l, axes=0)
        prj_l2k_t2f_lhs_inv: ndarray = np.linalg.inv(prj_l2k_t2f_lhs)
        prj_l2k_t2f: ndarray = prj_l2k_t2f_lhs_inv @ prj_l2k_t2f_rhs
        m = np.eye(_fk, dtype=real)
        for _i in range(_dx):
            interop: ndarray = np.zeros((_cl, _es), dtype=real)
            stabilization_op33 = np.zeros((_fk, _es), dtype=real)
            _ci = _i * _cl
            _cj = (_i + 1) * _cl
            interop[:, _ci:_cj] += np.eye(_cl, dtype=real)
            # that_01: ndarray = prj_l2k_t2f @ (- prj_r2l_t2t @ recop[_cri: _crj, :] + interop)
            that_01: ndarray = prj_l2k_t2f @ interop
            that: ndarray = that_01
            stabilization_op33 -= that
            _ci = _dx * _cl + _f * _dx * _fk + _i * _fk
            _cj = _dx * _cl + _f * _dx * _fk + (_i + 1) * _fk
            stabilization_op33[:, _ci:_cj] += m
            stabilization_operator += (1.0 / h_f) * stabilization_op33.T @ m_mas_f @ stabilization_op33
    return stabilization_operator

def get_stabilization_operator(field: Field, finite_element: FiniteElement, cell: Shape,
                                   faces: List[Shape]) -> ndarray:
    if field.derivation_type in [DerivationType.LARGE_STRAIN_AXISYMMETRIC, DerivationType.SMALL_STRAIN_AXISYMMETRIC]:
        return get_stabilization_operator3hho_axisymmetric(field, finite_element, cell, faces)
        # return get_hdg_stabilization_axisymmetric(field, finite_element, cell, faces)
    else:
        return get_stabilization_operator3hho(field, finite_element, cell, faces)