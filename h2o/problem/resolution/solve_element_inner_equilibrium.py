from h2o.problem.problem import Problem, clean_res_dir
from h2o.problem.material import Material
from h2o.fem.element.element import Element
from h2o.h2o import *
from typing import TextIO

from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sys


def process_cell_quadrature_points(
        problem: Problem,
        material: Material,
        element: Element,
        _element_index: int,
        local_faces_unknown_vector: ndarray,
        _dt: float,
        outfile: TextIO
) -> bool:
    x_c: ndarray = element.cell.get_centroid()
    bdc: ndarray = element.cell.get_bounding_box()
    _nf: int = len(element.faces)
    _dx: int = problem.field.field_dimension
    _fk: int = problem.finite_element.face_basis_k.dimension
    _cl: int = problem.finite_element.cell_basis_l.dimension
    _c0_c: int = _dx * _cl
    # --- INITIALIZE MATRIX AND VECTORS
    element_stiffness_matrix = np.zeros((element.element_size, element.element_size), dtype=real)
    element_internal_forces = np.zeros((element.element_size,), dtype=real)
    element_external_forces = np.zeros((element.element_size,), dtype=real)
    _io: int = problem.finite_element.computation_integration_order
    cell_quadrature_size = element.cell.get_quadrature_size(
        _io, quadrature_type=problem.quadrature_type
    )
    cell_quadrature_points = element.cell.get_quadrature_points(
        _io, quadrature_type=problem.quadrature_type
    )
    cell_quadrature_weights = element.cell.get_quadrature_weights(
        _io, quadrature_type=problem.quadrature_type
    )
    # --- ITERATING OVER QUADRATURE POINTS
    for _qc in range(cell_quadrature_size):
        _qp = element.quad_p_indices[_qc]
        _w_q_c = cell_quadrature_weights[_qc]
        _x_q_c = cell_quadrature_points[:, _qc]
        # --- COMPUTE STRAINS AND SET THEM IN THE BEHAVIOUR LAW
        transformation_gradient = element.get_transformation_gradient(local_faces_unknown_vector, _qc)
        material.mat_data.s1.gradients[_qp] = transformation_gradient
        # --- INTEGRATE BEHAVIOUR LAW
        integ_res = mgis_bv.integrate(material.mat_data, material.integration_type, _dt, _qp, (_qp + 1))
        if integ_res != 1:
            print("++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
            print("++++++++++++++++ - POINT {}".format(_x_q_c))
            print("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
            outfile.write(
                "++++++++++++++++ @ CELL : {} | INTEG_RES FAILURE @ QUAD POINT {}".format(_element_index, _qp))
            outfile.write("\n")
            outfile.write("++++++++++++++++ - POINT {}".format(_x_q_c))
            outfile.write("\n")
            outfile.write("++++++++++++++++ - STRAIN {}".format(transformation_gradient))
            outfile.write("\n")
            return False
        else:
            # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
            element_stiffness_matrix += _w_q_c * (
                    element.gradients_operators[_qc].T @ material.mat_data.K[_qp] @
                    element.gradients_operators[_qc]
            )
            # print("CEP : \n{}".format(material.mat_data.K[_qp]))
            # --- COMPUTE STIFFNESS MATRIX CONTRIBUTION AT QUADRATURE POINT
            element_internal_forces += _w_q_c * (
                    element.gradients_operators[_qc].T @ material.mat_data.s1.thermodynamic_forces[_qp]
            )
    return True

def
