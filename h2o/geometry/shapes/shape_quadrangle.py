from h2o.geometry.geometry import *
import h2o.geometry.shapes.shape_triangle as shape_triangle
import h2o.geometry.quadratures.gauss.gauss_quadrangle2 as gauss_quadrangle
from h2o.geometry.quadratures.quadrature import *


def get_quadrangle_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_barycenter(vertices)


def check_quadrangle_vertices_consistency(vertices: ndarray):
    """

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if number_of_vertices != 4:
        raise GeometryError("a quadrangle is defined by 4 points, not {}".format(number_of_vertices))
    if euclidean_dimension < 2 or euclidean_dimension > 3:
        raise GeometryError("a quadrangle is defined in 2 or 3 dimension, not {}".format(euclidean_dimension))
    check_points_non_alignment(vertices)
    if euclidean_dimension == 3:
        check_points_coplanar(vertices)


def get_quadrangle_diagonals(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    d_0 = vertices[:, 2] - vertices[:, 0]
    d_1 = vertices[:, 3] - vertices[:, 1]
    diagonals = np.array([d_0, d_1]).T
    return diagonals


def get_quadrangle_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    diagonals = get_quadrangle_diagonals(vertices)
    quadrangle_diameter = np.max([np.linalg.norm(diagonals[:, 0]), np.linalg.norm(diagonals[:, 1])])
    return quadrangle_diameter


def get_quadrangle_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    diagonals = get_quadrangle_diagonals(vertices)
    if euclidean_dimension == 3:
        p = get_quadrangle_rotation_matrix(vertices)
        dprime = (p @ diagonals.T).T
    elif euclidean_dimension == 2:
        dprime = diagonals
    else:
        raise GeometryError("euclidean dimension must be either 2 or 3, not {}".format(euclidean_dimension))
    d_0 = dprime[:, 0]
    d_1 = dprime[:, 1]
    quadrangle_area = (1.0 / 2.0) * np.abs(d_0[0] * d_1[1] - d_0[1] * d_1[0])
    return quadrangle_area


def get_quadrangle_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    quadrangle_centroid = get_quadrangle_barycenter(vertices)
    return quadrangle_centroid


def get_quadrangle_rotation_matrix(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 3:
        tri = vertices[:, :-1]
        p = shape_triangle.get_triangle_rotation_matrix(tri)
    else:
        raise GeometryError("a triangle is a face only if the euclidean dimension is 3, not {}".format(euclidean_dimension))
    return p


def get_quadrangle_quadrature_size(
    integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> int:
    """

    Args:
        integration_order:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_quadrangle.get_number_of_quadrature_points_in_quadrangle(integration_order)
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_size


def get_quadrangle_quadrature_weights(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    # projected_vertices = (rotation_matrix @ vertices)[:-1]
    if euclidean_dimension == 2:
        projected_vertices = vertices
    elif euclidean_dimension == 3:
        rotation_matrix = get_quadrangle_rotation_matrix(vertices)
        projected_vertices = (rotation_matrix @ vertices)[:-1]
    else:
        raise GeometryError("euclidean dimension must be either 2 or 3, not {}".format(euclidean_dimension))
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_quadrangle.get_number_of_quadrature_points_in_quadrangle(integration_order)
        quadrature_reference_weights = gauss_quadrangle.get_reference_quadrangle_quadrature_item(
            integration_order, QuadratureItem.WEIGHTS
        )
        jacobian_weights = np.zeros((quadrature_size,), dtype=real)
        jacobian_operators = gauss_quadrangle.get_reference_quadrangle_quadrature_item(
            integration_order, QuadratureItem.JACOBIAN
        )
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    for i, jacobian_operator in enumerate(jacobian_operators):
        jacobian = np.zeros((2, 2), dtype=real)
        jacobian[0, 0] = jacobian_operator[0] @ projected_vertices[0, :]
        jacobian[0, 1] = jacobian_operator[1] @ projected_vertices[0, :]
        jacobian[1, 0] = jacobian_operator[2] @ projected_vertices[1, :]
        jacobian[1, 1] = jacobian_operator[3] @ projected_vertices[1, :]
        jacobian_weights[i] = np.abs(np.linalg.det(jacobian))
        # jacobian_weights[i] = np.linalg.det(jacobian)
        if jacobian_weights[i] < 1.e-12:
            print("jacobian det value is : {}".format(jacobian_weights[i]))
    quadrature_weights = quadrature_reference_weights * jacobian_weights
    return quadrature_weights


def get_quadrangle_quadrature_points(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_reference_points = gauss_quadrangle.get_reference_quadrangle_quadrature_item(
            integration_order, QuadratureItem.POINTS
        )
        quadrature_points = (quadrature_reference_points @ vertices.T).T
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_points
