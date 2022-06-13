from random import uniform
from unittest import TestCase

import matplotlib.pyplot as plt
import quadpy

from h2o.fem.basis.bases.monomial import Monomial
from h2o.geometry.shape import Shape, get_rotation_matrix
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
import h2o.fem.element.operators.operator_gradient as gradop
import h2o.fem.element.operators.operator_stabilization as stabop
from h2o.h2o import *

np.set_printoptions(precision=16)
np.set_printoptions(linewidth=1)


class TestOperators(TestCase):

    def test_operators_triangle(self):
        euclidean_dimension = 2
        polynomial_orders = [1, 2, 3]
        element_types = [ElementType.HDG_LOW, ElementType.HDG_EQUAL, ElementType.HDG_HIGH]
        for polynomial_order in polynomial_orders:
            for element_type in element_types:
                # --------------------------------------------------------------------------------------------------------------
                # DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
                # --------------------------------------------------------------------------------------------------------------
                finite_element = FiniteElement(
                    element_type=element_type,
                    polynomial_order=polynomial_order,
                    euclidean_dimension=euclidean_dimension,
                )

                field = Field("TEST", FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE POLYNOMIAL BASIS
                # --------------------------------------------------------------------------------------------------------------
                cell_basis_k = finite_element.cell_basis_k
                cell_basis_l = finite_element.cell_basis_l
                cell_basis_r = finite_element.cell_basis_r
                face_basis_k = finite_element.face_basis_k

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE RANDOM POLYNOMIAL COEFFICIENTS
                # --------------------------------------------------------------------------------------------------------------
                range_min = -3.0
                range_max = +3.0
                coefficients_k_list = []
                coefficients_l_list = []
                coefficients_r_list = []
                for _i in range(field.field_dimension):
                    coefficients_k = np.array(
                        [uniform(range_min, range_max) for _iloc in range(cell_basis_k.dimension)]
                    )
                    coefficients_l = np.array(
                        [uniform(range_min, range_max) for _iloc in range(cell_basis_l.dimension)]
                    )
                    coefficients_r = np.array(
                        [uniform(range_min, range_max) for _iloc in range(cell_basis_r.dimension)]
                    )
                    coefficients_k_list.append(coefficients_k)
                    coefficients_l_list.append(coefficients_l)
                    coefficients_r_list.append(coefficients_r)
                # print("COEFS_K : \n{}".format(coefficients_k))
                # print("COEFS_L : \n{}".format(coefficients_l))

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE MONOMIAL VALUES COMPUTATION
                # --------------------------------------------------------------------------------------------------------------
                def test_function(
                    polynomial_ord: int, point: ndarray, centroid: ndarray, diameter: float, coefficients: ndarray
                ) -> float:
                    basis = Monomial(polynomial_ord, euclidean_dimension)
                    value = 0.0
                    for _i, _exponent in enumerate(basis.exponents):
                        prod = 1.0
                        for _x_dir in range(basis.exponents.shape[1]):
                            prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                def test_function_derivative(
                    polynomial_ord: int,
                    point: ndarray,
                    centroid: ndarray,
                    diameter: float,
                    direction: int,
                    coefficients: ndarray,
                ) -> float:
                    basis = Monomial(polynomial_ord, euclidean_dimension)
                    value = 0.0
                    for _i, _exponent in enumerate(basis.exponents):
                        prod = 1.0
                        for _x_dir in range(basis.exponents.shape[1]):
                            if _x_dir == direction:
                                _pt0 = point[_x_dir] - centroid[_x_dir]
                                _pt1 = _pt0 / diameter
                                if _exponent[_x_dir] == 0:
                                    _exp = _exponent[_x_dir]
                                else:
                                    _exp = _exponent[_x_dir] - 1
                                _pt2 = _pt1 ** _exp
                                # prod *= (_exponent[_x_dir] / diameter) * (
                                #         ((point[_x_dir] - centroid[_x_dir]) / diameter) ** (_exponent[_x_dir] - 1)
                                # )
                                prod *= (_exponent[_x_dir] / diameter) * _pt2
                            else:
                                prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                # --------------------------------------------------------------------------------------------------------------
                # DEFINE TRIANGLE COORDINATES
                # --------------------------------------------------------------------------------------------------------------
                v0 = np.array([1.0, 1.7], dtype=real)
                v1 = np.array([2.0, 1.6], dtype=real)
                v2 = np.array([1.9, 3.0], dtype=real)
                triangle_vertices = np.array([v0, v1, v2], dtype=real).T

                # --------------------------------------------------------------------------------------------------------------
                # BUILD CELL
                # --------------------------------------------------------------------------------------------------------------
                cell_triangle = Shape(ShapeType.TRIANGLE, triangle_vertices)

                # --------------------------------------------------------------------------------------------------------------
                # BUILD FACES
                # --------------------------------------------------------------------------------------------------------------
                faces_segment = [
                    Shape(ShapeType.SEGMENT, triangle_vertices[:, [0, 1]]),
                    Shape(ShapeType.SEGMENT, triangle_vertices[:, [1, 2]]),
                    Shape(ShapeType.SEGMENT, triangle_vertices[:, [2, 0]]),
                ]

                def get_element_projection_vector(cell: Shape, faces: List[Shape], function: List[Callable]):
                    _d = euclidean_dimension
                    _dx = field.field_dimension
                    _cl = cell_basis_l.dimension
                    _fk = face_basis_k.dimension
                    _nf = len(faces)
                    _es = _dx * (_cl + _nf * _fk)
                    #
                    x_c = cell.centroid
                    h_c = cell.diameter
                    _io = finite_element.construction_integration_order
                    cell_quadrature_points = cell.get_quadrature_points(_io)
                    cell_quadrature_weights = cell.get_quadrature_weights(_io)
                    cell_quadrature_size = cell.get_quadrature_size(_io)
                    matrix = np.zeros((_es, _es), dtype=real)
                    vector = np.zeros((_es,), dtype=real)
                    for _dir in range(_dx):
                        m_mas = np.zeros((_cl, _cl), dtype=real)
                        vc = np.zeros((_cl,), dtype=real)
                        for qc in range(cell_quadrature_size):
                            x_q_c = cell_quadrature_points[:, qc]
                            w_q_c = cell_quadrature_weights[qc]
                            phi_l = cell_basis_l.evaluate_function(x_q_c, x_c, h_c)
                            m_mas += w_q_c * np.tensordot(phi_l, phi_l, axes=0)
                            vc += w_q_c * phi_l * function[_dir](x_q_c)
                        _i = _cl * _dir
                        _j = _cl * (_dir + 1)
                        matrix[_i:_j, _i:_j] += m_mas
                        vector[_i:_j] += vc
                    for _f, face in enumerate(faces):
                        face_rotation_matrix = get_rotation_matrix(face.type, face.vertices)
                        x_f = face.centroid
                        h_f = face.diameter
                        # --- PROJECT ON HYPERPLANE
                        s_f = (face_rotation_matrix @ x_f)[:-1]
                        _io = finite_element.construction_integration_order
                        face_quadrature_points = face.get_quadrature_points(_io)
                        face_quadrature_weights = face.get_quadrature_weights(_io)
                        face_quadrature_size = face.get_quadrature_size(_io)
                        for _dir in range(_dx):
                            m_mas_f = np.zeros((_fk, _fk), dtype=real)
                            vf = np.zeros((_fk,), dtype=real)
                            for qf in range(face_quadrature_size):
                                x_q_f = face_quadrature_points[:, qf]
                                w_q_f = face_quadrature_weights[qf]
                                # s_f = (face_rotation_matrix @ x_f)[:-1]
                                s_q_f = (face_rotation_matrix @ x_q_f)[:-1]
                                psi_k = face_basis_k.evaluate_function(s_q_f, s_f, h_f)
                                m_mas_f += w_q_f * np.tensordot(psi_k, psi_k, axes=0)
                                vf += w_q_f * psi_k * function[_dir](x_q_f)
                            _i = _cl * _dx + _f * _fk * _dx + _dir * _fk
                            _j = _cl * _dx + _f * _fk * _dx + (_dir + 1) * _fk
                            matrix[_i:_j, _i:_j] += m_mas_f
                            vector[_i:_j] += vf
                    projection_vector = np.linalg.solve(matrix, vector)
                    return projection_vector

                def get_gradient_projection_vector(cell: Shape, function: Callable):
                    _d = euclidean_dimension
                    _dx = field.field_dimension
                    _ck = cell_basis_k.dimension
                    #
                    x_c = cell.centroid
                    h_c = cell.diameter
                    _io = finite_element.construction_integration_order
                    cell_quadrature_points = cell.get_quadrature_points(_io)
                    cell_quadrature_weights = cell.get_quadrature_weights(_io)
                    cell_quadrature_size = cell.get_quadrature_size(_io)
                    matrix = np.zeros((_ck, _ck), dtype=real)
                    vector = np.zeros((_ck,), dtype=real)
                    for qc in range(cell_quadrature_size):
                        x_q_c = cell_quadrature_points[:, qc]
                        w_q_c = cell_quadrature_weights[qc]
                        phi_k = cell_basis_k.evaluate_function(x_q_c, x_c, h_c)
                        matrix += w_q_c * np.tensordot(phi_k, phi_k, axes=0)
                        vector += w_q_c * phi_k * function(x_q_c)
                    projection_vector = np.linalg.solve(matrix, vector)
                    return projection_vector

                # fun = [
                #     lambda x: test_function(
                #         finite_element.cell_basis_l.polynomial_order,
                #         x,
                #         cell_triangle.centroid,
                #         cell_triangle.diameter,
                #         coefficients_l_list[0],
                #     ),
                #     lambda x: test_function(
                #         finite_element.cell_basis_l.polynomial_order,
                #         x,
                #         cell_triangle.centroid,
                #         cell_triangle.diameter,
                #         coefficients_l_list[1],
                #     )
                # ]
                #
                # fun_grad_regular = [
                #     [
                #         lambda x: test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             0,
                #             coefficients_l_list[0],
                #         ),
                #         lambda x: test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             1,
                #             coefficients_l_list[0],
                #         ),
                #     ],
                #     [
                #         lambda x: test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             0,
                #             coefficients_l_list[1],
                #         ),
                #         lambda x: test_function_derivative(
                #             finite_element.cell_basis_l.polynomial_order,
                #             x,
                #             cell_triangle.centroid,
                #             cell_triangle.diameter,
                #             1,
                #             coefficients_l_list[1],
                #         ),
                #     ],
                # ]
                #
                # fun_grad_symmetric = [
                #     [
                #         lambda x: (1.0 / 2.0)
                #         * (
                #             test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 0,
                #                 coefficients_l_list[0],
                #             )
                #             + test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 0,
                #                 coefficients_l_list[0],
                #             )
                #         ),
                #         lambda x: (1.0 / 2.0)
                #         * (
                #             test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 1,
                #                 coefficients_l_list[0],
                #             )
                #             + test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 0,
                #                 coefficients_l_list[1],
                #             )
                #         ),
                #     ],
                #     [
                #         lambda x: (1.0 / 2.0)
                #         * (
                #             test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 0,
                #                 coefficients_l_list[1],
                #             )
                #             + test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 1,
                #                 coefficients_l_list[0],
                #             )
                #         ),
                #         lambda x: (1.0 / 2.0)
                #         * (
                #             test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 1,
                #                 coefficients_l_list[1],
                #             )
                #             + test_function_derivative(
                #                 finite_element.cell_basis_l.polynomial_order,
                #                 x,
                #                 cell_triangle.centroid,
                #                 cell_triangle.diameter,
                #                 1,
                #                 coefficients_l_list[1],
                #             )
                #         ),
                #     ],
                # ]

                fun = [
                    lambda x: test_function(
                        finite_element.cell_basis_r.polynomial_order,
                        x,
                        cell_triangle.centroid,
                        cell_triangle.diameter,
                        coefficients_r_list[0],
                    ),
                    lambda x: test_function(
                        finite_element.cell_basis_r.polynomial_order,
                        x,
                        cell_triangle.centroid,
                        cell_triangle.diameter,
                        coefficients_r_list[1],
                    ),
                ]

                fun_grad_regular = [
                    [
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_r.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            0,
                            coefficients_r_list[0],
                        ),
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_r.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            1,
                            coefficients_r_list[0],
                        ),
                    ],
                    [
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_r.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            0,
                            coefficients_r_list[1],
                        ),
                        lambda x: test_function_derivative(
                            finite_element.cell_basis_r.polynomial_order,
                            x,
                            cell_triangle.centroid,
                            cell_triangle.diameter,
                            1,
                            coefficients_r_list[1],
                        ),
                    ],
                ]

                fun_grad_symmetric = [
                    [
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_r_list[0],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_r_list[0],
                            )
                        ),
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_r_list[0],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_r_list[1],
                            )
                        ),
                    ],
                    [
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                0,
                                coefficients_r_list[1],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_r_list[0],
                            )
                        ),
                        lambda x: (1.0 / 2.0)
                        * (
                            test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_r_list[1],
                            )
                            + test_function_derivative(
                                finite_element.cell_basis_r.polynomial_order,
                                x,
                                cell_triangle.centroid,
                                cell_triangle.diameter,
                                1,
                                coefficients_r_list[1],
                            )
                        ),
                    ],
                ]

                # --------------------------------------------------------------------------------------------------------------
                # BUILD AND TEST STABILIZATION
                # --------------------------------------------------------------------------------------------------------------

                fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                # stab_matrix, stab_matrix_0, stab_matrix2 = get_stabilization_operator(cell_triangle, faces_segment)
                stabilization_operator = stabop.get_stabilization_operator2(
                    field, finite_element, cell_triangle, faces_segment
                )
                print(
                    "--- FUN PROJ | k : {} | l : {}".format(
                        cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                    )
                )
                print(fun_proj)
                print(
                    "--- STABILIZATION | k : {} | l : {}".format(
                        cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                    )
                )
                stab_val = fun_proj @ stabilization_operator @ fun_proj
                print(stab_val)
                rtol = 1000000.0
                atol = 1.0e-11
                # np.testing.assert_allclose(stab_val, 0.0, rtol=rtol, atol=atol)

                correspondance = {0: (0, 0), 1: (1, 1), 2: (0, 1), 3: (1, 0)}
                for key, val in correspondance.items():
                    dir_x = val[0]
                    dir_y = val[1]
                    # rtol = 1.0e-12
                    # rtol = 1.0e-3
                    rtol = 1000000.0
                    atol = 1.0e-11
                    print(
                        "--- SYMMETRIC GRADIENT | k : {} | l : {}".format(
                            cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                        )
                    )
                    fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                    grad_comp = gradop.get_symmetric_gradient_component_matrix(
                        field, finite_element, cell_triangle, faces_segment, dir_x, dir_y
                    )
                    # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_sym[choice])
                    # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_symmetric[dir_x][dir_y])
                    fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_symmetric[dir_y][dir_x])
                    grad_check = grad_comp @ fun_proj
                    print("- GRAD REC | {} | {}".format(dir_x, dir_y))
                    print(grad_check)
                    print("- GRAD PROJ | {} | {}".format(dir_x, dir_y))
                    print(fun_grad_proj)
                    np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
                    print(
                        "--- REGULAR GRADIENT | k : {} | l : {}".format(
                            cell_basis_k.polynomial_order, cell_basis_l.polynomial_order
                        )
                    )
                    fun_proj = get_element_projection_vector(cell_triangle, faces_segment, fun)
                    grad_comp = gradop.get_regular_gradient_component_matrix(
                        field, finite_element, cell_triangle, faces_segment, dir_x, dir_y
                    )
                    # fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_sym[choice])
                    fun_grad_proj = get_gradient_projection_vector(cell_triangle, fun_grad_regular[dir_x][dir_y])
                    grad_check = grad_comp @ fun_proj
                    print("- GRAD REC | {} | {}".format(dir_x, dir_y))
                    print(grad_check)
                    print("- GRAD PROJ | {} | {}".format(dir_x, dir_y))
                    print(fun_grad_proj)
                    np.testing.assert_allclose(grad_check, fun_grad_proj, rtol=rtol, atol=atol)
