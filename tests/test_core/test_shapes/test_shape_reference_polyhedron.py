from random import uniform
from unittest import TestCase
from scipy import integrate

import matplotlib.pyplot as plt

import quadpy

from h2o.fem.basis.bases.monomial import Monomial
from h2o.geometry.shape import Shape
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *

np.set_printoptions(precision=16)
np.set_printoptions(linewidth=1)

# val ref is (3.924576491877647, 4.80971142793224e-08) of integration with scipy


class TestReferencePolyhederon(TestCase):
    def test_reference_polyhedron_geometry_cell(self, verbose=True):
        """

          V8  o_______________o V7
             /|     V6       /|
            / |     o       / |
        V4 o_______________o V5
           |  |            |  |
        V3 |  o___________ |__o V2              Z
           | /             | /                  ^ Y
           |/              |/                   |/
        V0 o_______________o V1                 0---> X

        Args:
            verbose:
        """
        v0 = np.array([0.0, 0.0, 0.0], dtype=real)
        v1 = np.array([1.0, 0.0, 0.0], dtype=real)
        v2 = np.array([1.0, 1.0, 0.0], dtype=real)
        v3 = np.array([0.0, 1.0, 0.0], dtype=real)
        v4 = np.array([0.0, 0.0, 1.0], dtype=real)
        v5 = np.array([1.0, 0.0, 1.0], dtype=real)
        v6 = np.array([0.5, 0.5, 1.0], dtype=real)
        v7 = np.array([1.0, 1.0, 1.0], dtype=real)
        v8 = np.array([0.0, 1.0, 1.0], dtype=real)
        vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8], dtype=real).T
        # --- CONNECTIVITY, COUNTER CLOCK WISE
        # connectivity = [
        #     [0, 3, 2, 1],
        #     [0, 1, 5, 4],
        #     [1, 2, 7, 5],
        #     [2, 3, 8, 7],
        #     [3, 0, 4, 8],
        #     [4, 5, 6],
        #     [5, 7, 6],
        #     [7, 8, 6],
        #     [8, 4, 6],
        # ]
        # --- CONNECTIVIY, CLOCK WISE
        connectivity = [
            [0, 1, 2, 3],
            [0, 4, 5, 1],
            [1, 5, 7, 2],
            [2, 7, 8, 3],
            [3, 8, 4, 0],
            [6, 5, 4],
            [6, 7, 5],
            [6, 8, 7],
            [6, 4, 8],
        ]
        shape = Shape(ShapeType.POLYHEDRON, vertices, connectivity=connectivity)
        x_c = shape.get_centroid()
        h_c = shape.get_diameter()
        v_c = shape.get_volume()
        if verbose:
            print("-- centroid :\n{}".format(x_c))
            print("-- diameter :\n{}".format(h_c))
            print("-- volume :\n{}".format(v_c))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_c[0], x_c[1], x_c[2], c="b", marker="o")
            for i in range(vertices.shape[1]):
                ax.scatter(vertices[0, i], vertices[1, i], vertices[2, i], c="b", marker="o")
            plt.show()

    def test_reference_polyhedron_quadrature(self, verbose=True):
        """

          V8  o_______________o V7
             /|     V6       /|
            / |     o       / |
        V4 o_______________o V5
           |  |            |  |
        V3 |  o___________ |__o V2              Z
           | /             | /                  ^ Y
           |/              |/                   |/
        V0 o_______________o V1                 0---> X

        Args:
            verbose:
        """
        v0 = np.array([0.0, 0.0, 0.0], dtype=real)
        v1 = np.array([1.0, 0.0, 0.0], dtype=real)
        v2 = np.array([1.0, 1.0, 0.0], dtype=real)
        v3 = np.array([0.0, 1.0, 0.0], dtype=real)
        v4 = np.array([0.0, 0.0, 1.0], dtype=real)
        v5 = np.array([1.0, 0.0, 1.0], dtype=real)
        v6 = np.array([0.5, 0.5, 1.0], dtype=real)
        v7 = np.array([1.0, 1.0, 1.0], dtype=real)
        v8 = np.array([0.0, 1.0, 1.0], dtype=real)
        vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8], dtype=real).T
        # --- CONNECTIVITY, COUNTER CLOCK WISE
        # connectivity = [
        #     [0, 3, 2, 1],
        #     [0, 1, 5, 4],
        #     [1, 2, 7, 5],
        #     [2, 3, 8, 7],
        #     [3, 0, 4, 8],
        #     [4, 5, 6],
        #     [5, 7, 6],
        #     [7, 8, 6],
        #     [8, 4, 6],
        # ]
        # --- CONNECTIVIY, CLOCK WISE
        connectivity = [
            [0, 1, 2, 3],
            [0, 4, 5, 1],
            [1, 5, 7, 2],
            [2, 7, 8, 3],
            [3, 8, 4, 0],
            [6, 5, 4],
            [6, 7, 5],
            [6, 8, 7],
            [6, 4, 8],
        ]
        shape = Shape(ShapeType.POLYHEDRON, vertices, connectivity=connectivity)
        for _io in range(1, 9):
            # --- DECLARE FUNCTION
            f = lambda x: np.exp(x[0]) * np.sin(x[0] * x[1] / (x[2] + 0.003)) + x[1] * x[2] + 3.0
            f_scipy = lambda x, y, z: np.exp(x) * np.sin(x * y / (z + 0.003)) + y * z + 3.0
            # --- GET QUADPY ESTIMATION
            scheme = quadpy.c3.get_good_scheme(_io)
            val = scheme.integrate(f, [[[v0, v4], [v3, v8]], [[v1, v5], [v2, v7]]])
            # --- GET H20 ESTIMATION
            val_num = 0.0
            shape_quadrature_points = shape.get_quadrature_points(_io)
            shape_quadrature_weights = shape.get_quadrature_weights(_io)
            shape_quadrature_size = shape.get_quadrature_size(_io)
            for _qc in range(shape_quadrature_size):
                x_qc = shape_quadrature_points[:, _qc]
                w_qc = shape_quadrature_weights[_qc]
                val_num += w_qc * f(x_qc)
            # --- GET SCIPY ESTIMATION
            # val_scp = integrate.tplquad(
            #     f_scipy, 0.0, 1.0, lambda x: 0.0, lambda x: 1.0, lambda x, y: 0.0, lambda x, y: 1.0
            # )
            if verbose:
                print("-- integration order : {}".format(_io))
                print("val_h2O : {}".format(val_num))
                print("val_qud : {}".format(val))
                # print("val_scp : {}".format(val_scp))
                x_c = shape.get_centroid()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(x_c[0], x_c[1], x_c[2], c="b", marker="o")
                for i in range(vertices.shape[1]):
                    ax.scatter(vertices[0, i], vertices[1, i], vertices[2, i], c="b", marker="o")
                for _qc in range(shape_quadrature_size):
                    _x_qc = shape_quadrature_points[:, _qc]
                    ax.scatter(_x_qc[0], _x_qc[1], _x_qc[2], c="g")
                plt.show()

    def test_reference_polyhedron_cell(self, verbose=True):
        """

             V3 o________________o V2
               /|               /|
              / |              / |
             /  |             /  |                Y
            /   |            /   |                ^
        V7 o________________o V6 |                |
           |    |           |    |                0---> X
           | V0 o___________|____o V1            /
           |   /            |   /                Z
           |  /             |  /
           | /              | /
           |/               |/
        V4 o________________o V5


             V3 o________________o V2
               /|               /|
              / |              / |
             /  |             /  |                Y
            /   |            /   |                ^
        V8 o________________o V7 |                |
           |    |           |    |                0---> X
           | V0 o___________|____o V1            /
           |   /            |   /                Z
           |  /    o V6     |  /
           | /              | /
           |/               |/
        V4 o________________o V5


            V2 o
               |\..
               |   \..
               |      \..
               |         \..
               |            \..
               |               \
               o_V0_____________o V1
          |   /           ..../
             /       ..../
          / /   ..../
          :/.../
        V2 o


        Args:
            verbose:

        Returns:

        """
        euclidean_dimension = 3
        polynomial_orders = [1, 2, 3]
        element_types = [ElementType.HDG_LOW, ElementType.HDG_EQUAL, ElementType.HDG_HIGH]
        for polynomial_order in polynomial_orders:
            for element_type in element_types:
                # --- DEFINE POLYNOMIAL ORDER AND INTEGRATION ORDER
                finite_element = FiniteElement(
                    element_type=element_type,
                    polynomial_order=polynomial_order,
                    euclidean_dimension=euclidean_dimension,
                )

                # --- DEFINE POLYNOMIAL BASIS
                cell_basis_k = finite_element.cell_basis_k
                cell_basis_l = finite_element.cell_basis_l

                # --- DEFINE RANDOM POLYNOMIAL COEFFICIENTS
                range_min = -3.0
                range_max = +3.0
                coefficients_k = np.array([uniform(range_min, range_max) for _i in range(cell_basis_k.dimension)])
                coefficients_l = np.array([uniform(range_min, range_max) for _i in range(cell_basis_l.dimension)])
                if verbose:
                    print("COEFS_K : \n{}".format(coefficients_k))
                    print("COEFS_L : \n{}".format(coefficients_l))

                # --- DEFINE MONOMIAL VALUES COMPUTATION
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
                                # _pt0 = point[_x_dir] - centroid[_x_dir]
                                # _pt1 = _pt0 / diameter
                                # if _exponent[_x_dir] == 0:
                                #     _exp = _exponent[_x_dir]
                                # else:
                                #     _exp = _exponent[_x_dir] - 1
                                # _pt2 = _pt1 ** _exp
                                if _exponent[_x_dir] != 0:
                                    prod *= (_exponent[_x_dir] / diameter) * (
                                        ((point[_x_dir] - centroid[_x_dir]) / diameter) ** (_exponent[_x_dir] - 1)
                                    )
                                else:
                                    prod *= 0.0
                                # prod *= (_exponent[_x_dir] / diameter) * _pt2
                            else:
                                prod *= ((point[_x_dir] - centroid[_x_dir]) / diameter) ** _exponent[_x_dir]
                        prod *= coefficients[_i]
                        value += prod
                    return value

                # --- DEFINE TRIANGLE COORDINATES
                v0 = np.array([0.0, 0.0, 0.0], dtype=real)
                v1 = np.array([1.0, 0.0, 0.0], dtype=real)
                v2 = np.array([1.0, 1.0, 0.0], dtype=real)
                v3 = np.array([0.0, 1.0, 0.0], dtype=real)
                v4 = np.array([0.0, 0.0, 1.0], dtype=real)
                v5 = np.array([1.0, 0.0, 1.0], dtype=real)
                v6 = np.array([0.5, 0.5, 1.0], dtype=real)
                v7 = np.array([1.0, 1.0, 1.0], dtype=real)
                v8 = np.array([0.0, 1.0, 1.0], dtype=real)
                vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8], dtype=real).T
                # --- CONNECTIVITY, COUNTER CLOCK WISE
                # connectivity = [
                #     [0, 3, 2, 1],
                #     [0, 1, 5, 4],
                #     [1, 2, 7, 5],
                #     [2, 3, 8, 7],
                #     [3, 0, 4, 8],
                #     [4, 5, 6],
                #     [5, 7, 6],
                #     [7, 8, 6],
                #     [8, 4, 6],
                # ]
                # --- CONNECTIVIY, CLOCK WISE
                connectivity = [
                    [0, 1, 2, 3],
                    [0, 4, 5, 1],
                    [1, 5, 7, 2],
                    [2, 7, 8, 3],
                    [3, 8, 4, 0],
                    [6, 5, 4],
                    [6, 7, 5],
                    [6, 8, 7],
                    [6, 4, 8],
                ]
                # --- BUILD CELL
                shape = Shape(ShapeType.POLYHEDRON, vertices, connectivity=connectivity)
                x_c = shape.get_centroid()
                h_c = shape.get_diameter()
                _io = finite_element.construction_integration_order
                cell_quadrature_points = shape.get_quadrature_points(_io)
                cell_quadrature_weights = shape.get_quadrature_weights(_io)
                cell_quadrature_size = shape.get_quadrature_size(_io)

                # --- CHECK INTEGRATION IN CELL
                bases = [cell_basis_k, cell_basis_l]
                # orders = [face_polynomial_order, cell_polynomial_order]
                coefs = [coefficients_k, coefficients_l]
                # scheme = quadpy.t2.get_good_scheme(2 * finite_element.construction_integration_order)
                scheme = quadpy.c3.get_good_scheme(finite_element.construction_integration_order)
                for basis_0, coef_0 in zip(bases, coefs):
                    order_0 = basis_0.polynomial_order
                    for basis_1, coef_1 in zip(bases, coefs):
                        order_1 = basis_1.polynomial_order
                        for _i in range(euclidean_dimension):
                            for _j in range(euclidean_dimension):
                                mass_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                                stif_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                                advc_mat = np.zeros((basis_0.dimension, basis_1.dimension), dtype=real)
                                for _qc in range(cell_quadrature_size):
                                    _x_qc = cell_quadrature_points[:, _qc]
                                    _w_qc = cell_quadrature_weights[_qc]
                                    phi_0 = basis_0.evaluate_function(_x_qc, x_c, h_c)
                                    phi_1 = basis_1.evaluate_function(_x_qc, x_c, h_c)
                                    d_phi_0_i = basis_0.evaluate_derivative(_x_qc, x_c, h_c, _i)
                                    d_phi_1_j = basis_1.evaluate_derivative(_x_qc, x_c, h_c, _j)
                                    mass_mat += _w_qc * np.tensordot(phi_0, phi_1, axes=0)
                                    stif_mat += _w_qc * np.tensordot(d_phi_0_i, d_phi_1_j, axes=0)
                                    advc_mat += _w_qc * np.tensordot(phi_0, d_phi_1_j, axes=0)
                                mass_integral = coef_0 @ mass_mat @ coef_1
                                stif_integral = coef_0 @ stif_mat @ coef_1
                                advc_integral = coef_0 @ advc_mat @ coef_1
                                f_mass_check = lambda x: test_function(order_0, x, x_c, h_c, coef_0) * test_function(
                                    order_1, x, x_c, h_c, coef_1
                                )
                                f_stif_check = lambda x: test_function_derivative(
                                    order_0, x, x_c, h_c, _i, coef_0
                                ) * test_function_derivative(order_1, x, x_c, h_c, _j, coef_1)
                                f_advc_check = lambda x: test_function(
                                    order_0, x, x_c, h_c, coef_0
                                ) * test_function_derivative(order_1, x, x_c, h_c, _j, coef_1)
                                mass_integral_check = scheme.integrate(f_mass_check, [[[v0, v4], [v3, v8]], [[v1, v5], [v2, v7]]])
                                stif_integral_check = scheme.integrate(f_stif_check, [[[v0, v4], [v3, v8]], [[v1, v5], [v2, v7]]])
                                advc_integral_check = scheme.integrate(f_advc_check, [[[v0, v4], [v3, v8]], [[v1, v5], [v2, v7]]])
                                rtol = 1.0e-12
                                atol = 1.0e-12
                                if verbose:
                                    print(
                                        "MASS INTEGRAL CHECK | ORDER : {} | ELEM : {} | dir {}, {}, | order {}, {}".format(
                                            polynomial_order, element_type, _i, _j, order_0, order_1
                                        )
                                    )
                                    print("- QUADPY : {}".format(mass_integral_check))
                                    print("- H2O : {}".format(mass_integral))
                                    print(
                                        "STIFFNESS INTEGRAL CHECK | ORDER : {} | ELEM : {} | dir {}, {} | order {}, {}".format(
                                            polynomial_order, element_type, _i, _j, order_0, order_1
                                        )
                                    )
                                    print("- QUADPY : {}".format(stif_integral_check))
                                    print("- H2O : {}".format(stif_integral))
                                    print(
                                        "ADVECTION INTEGRAL CHECK | ORDER : {} | ELEM : {} | dir {}, {} | order {}, {}".format(
                                            polynomial_order, element_type, _i, _j, order_0, order_1
                                        )
                                    )
                                    print("- QUADPY : {}".format(advc_integral_check))
                                    print("- H2O : {}".format(advc_integral))
                                np.testing.assert_allclose(mass_integral_check, mass_integral, rtol=rtol, atol=atol)
                                np.testing.assert_allclose(stif_integral_check, stif_integral, rtol=rtol, atol=atol)
                                np.testing.assert_allclose(advc_integral_check, advc_integral, rtol=rtol, atol=atol)
