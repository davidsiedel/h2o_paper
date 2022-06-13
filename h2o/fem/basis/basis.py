from numpy import ndarray
from typing import Callable

import h2o.fem.basis.bases.monomial as imonomial
from h2o.h2o import *


class Basis:
    dimension: int
    polynomial_order: int
    evaluate_derivative: Callable[[ndarray, ndarray, ndarray, int], ndarray]
    evaluate_function: Callable[[ndarray, ndarray, ndarray], ndarray]

    def __init__(
        self, polynomial_order: int, euclidean_dimension: int, basis_type: BasisType = BasisType.MONOMIAL,
    ):
        """

        Args:
            polynomial_order:
            euclidean_dimension:
            basis_type:
        """
        if basis_type == BasisType.MONOMIAL:
            b = imonomial.Monomial(polynomial_order, euclidean_dimension)
        else:
            raise KeyError("unsupported basis")
        self.dimension = b.dimension
        self.polynomial_order = polynomial_order
        self.evaluate_function = b.get_phi_vector
        self.evaluate_derivative = b.get_d_phi_vector
