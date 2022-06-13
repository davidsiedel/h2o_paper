from numpy import ndarray
from typing import List
from h2o.h2o import *
from scipy.special import binom


def get_basis_dimension(polynomial_order: int, euclidean_dimension: int) -> int:
    """

    Args:
        polynomial_order:
        euclidean_dimension:

    Returns:

    """
    # basis_dimension = math.comb(polynomial_order + euclidean_dimension, polynomial_order,)
    basis_dimension = int(binom(polynomial_order + euclidean_dimension, polynomial_order))
    return basis_dimension


def get_exponents(polynomial_order: int, euclidean_dimension: int) -> ndarray:
    """

    Args:
        polynomial_order:
        euclidean_dimension:

    Returns:

    """
    basis_dimension = get_basis_dimension(polynomial_order, euclidean_dimension)
    n_rows = basis_dimension
    if euclidean_dimension > 0:
        n_cols = euclidean_dimension
    else:
        n_cols = 1
    exponents_matrix = np.zeros((n_rows, n_cols), dtype=np.uint8)
    if euclidean_dimension == 0:
        exponents_matrix[0, 0] = 0
    elif euclidean_dimension == 1:
        row_count = 0
        for s in range(polynomial_order + 1):
            exponents_matrix[s, 0] = s
            row_count += 1
    elif euclidean_dimension == 2:
        row_count = 0
        for s in range(polynomial_order + 1):
            for l in range(s + 1):
                m = s - l
                # exponents_matrix[row_count] = np.array([l, m], dtype=np.uint8)
                exponents_matrix[row_count] = np.array([m, l], dtype=np.uint8)
                row_count += 1
    elif euclidean_dimension == 3:
        row_count = 0
        for s in range(polynomial_order + 1):
            for l in range(s + 1):
                for m in range(s + 1):
                    if not (l + m) > s:
                        n = s - (l + m)
                        # exponents_matrix[row_count] = np.array([l, m, n], dtype=np.uint8)
                        exponents_matrix[row_count] = np.array([n, m, l], dtype=np.uint8)
                        row_count += 1
    else:
        raise ValueError("forbidden euclidean dimension")
    return exponents_matrix


def get_gradient_operator(polynomial_order: int, euclidean_dimension: int, dx: int) -> ndarray:
    """

    Args:
        polynomial_order:
        euclidean_dimension:
        dx:

    Returns:

    """
    basis_dimension = get_basis_dimension(polynomial_order, euclidean_dimension)
    gradient_operator = np.zeros((basis_dimension, basis_dimension), dtype=real)
    exponents = get_exponents(polynomial_order, euclidean_dimension)
    d_exponents_matrix = np.copy(exponents)
    d_exponents_matrix[:, dx] = d_exponents_matrix[:, dx] - np.ones(d_exponents_matrix[:, dx].shape, dtype=int,)
    for i, coef in enumerate(exponents[:, dx]):
        if not coef == 0:
            for j, e in enumerate(exponents):
                if (e == d_exponents_matrix[i]).all():
                    gradient_operator[i, j] = float(coef)
                    break
    return gradient_operator


class Monomial:
    dimension: int
    exponents: ndarray
    gradients: List[ndarray]

    def __init__(self, polynomial_order: int, euclidean_dimension: int) -> None:
        """

        Args:
            polynomial_order:
            euclidean_dimension:
        """
        self.dimension = get_basis_dimension(polynomial_order, euclidean_dimension)
        self.exponents = get_exponents(polynomial_order, euclidean_dimension)
        self.gradients = [
            get_gradient_operator(polynomial_order, euclidean_dimension, dx) for dx in range(euclidean_dimension)
        ]

    def get_phi_vector2(self, point: ndarray, centroid: ndarray, diameter: float) -> ndarray:
        """

        Args:
            point:
            centroid:
            diameter:

        Returns:

        """
        point_matrix = np.tile(point, (self.dimension, 1))
        centroid_matrix = np.tile(centroid, (self.dimension, 1))
        # phi_matrix = (2.0 * (point_matrix - centroid_matrix) / diameter) ** self.exponents
        _a0 = point_matrix - centroid_matrix
        _a1 = 2.0 * _a0 / diameter
        _a2 = _a1 ** self.exponents
        phi_vector = np.prod(_a2, axis=1, dtype=real)
        # phi_matrix = ((point_matrix - centroid_matrix) / diameter) ** self.exponents
        # phi_vector = np.prod(phi_matrix, axis=1, dtype=real)
        return phi_vector

    def get_phi_vector(self, point: ndarray, centroid: ndarray, bounding_box: ndarray) -> ndarray:
        """

        Args:
            point:
            centroid:
            diameter:

        Returns:

        """
        phi_vector = np.zeros((self.dimension,), dtype=real)
        euclidean_dimension = point.shape[0]
        for i in range(self.dimension):
            prod = 1.
            for j in range(euclidean_dimension):
                _a00 = (2.0 * (point[j] - centroid[j])/bounding_box[j]) ** self.exponents[i, j]
                prod *= _a00
            phi_vector[i] = prod
        return phi_vector

    def get_d_phi_vector2(self, point: ndarray, centroid: ndarray, diameter: float, dx: int) -> ndarray:
        """

        Args:
            point:
            centroid:
            diameter:
            dx:

        Returns:

        """
        grad_dx = self.gradients[dx]
        phi_vector = self.get_phi_vector(point, centroid, diameter)
        d_phi_vector = 2.0 * (1.0 / diameter) * (grad_dx @ phi_vector.T)
        return d_phi_vector

    def get_d_phi_vector(self, point: ndarray, centroid: ndarray, bounding_box: ndarray, dx: int) -> ndarray:
        """

        Args:
            point:
            centroid:
            diameter:
            dx:

        Returns:

        """
        d_phi_vector = np.zeros((self.dimension,), dtype=real)
        euclidean_dimension = point.shape[0]
        for i in range(self.dimension):
            prod = 1.
            for j in range(euclidean_dimension):
                if j != dx:
                    _a00 = (2.0 * (point[j] - centroid[j]) / bounding_box[j]) ** self.exponents[i, j]
                    prod *= _a00
                else:
                    if self.exponents[i, j] > 0:
                        _a00 = ((2.0 * self.exponents[i, j])/bounding_box[j]) * (2.0 * (point[j] - centroid[j])/bounding_box[j]) ** (self.exponents[i, j] - 1)
                        prod *= _a00
                    else:
                        prod *= 0.0
            d_phi_vector[i] = prod
        return d_phi_vector
