from h2o.h2o import *
from h2o.fem.element.finite_element import FiniteElement
from h2o.geometry.shape import Shape
from h2o.field.field import Field
import sys

np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=100, suppress=True,
                    threshold=sys.maxsize, formatter=None)


def get_identity_operators(field: Field, finite_element: FiniteElement, cell: Shape, faces: List[Shape]) -> ndarray:
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
    identity_operators = np.zeros((_c_is, _dx, _es), dtype=real)
    # --- CELL GEOMETRY
    bdc = cell.get_bounding_box()
    x_c = cell.get_centroid()
    identity_matrix = np.zeros((_dx, _es), dtype=real)
    for _qc in range(_c_is):
        x_qc = cell_quadrature_points[:, _qc]
        v_cl = finite_element.cell_basis_l.evaluate_function(x_qc, x_c, bdc)
        for _i in range(_dx):
            _c0 = _i * _cl
            _c1 = (_i + 1) * _cl
            identity_matrix[_i, _c0:_c1] = v_cl
        identity_operators[_qc] = identity_matrix
    return identity_operators
