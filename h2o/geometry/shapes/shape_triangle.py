from h2o.geometry.geometry import *
# import h2o.geometry.quadratures.gauss.gauss_triangle as gauss_triangle
import h2o.geometry.quadratures.quad2.tri as gauss_triangle
from h2o.geometry.quadratures.quadrature import *


def get_triangle_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_barycenter(vertices)


def get_triangle_edges(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    # e_0 = vertices[:, 1] - vertices[:, 0]
    # e_1 = vertices[:, 2] - vertices[:, 1]
    # e_2 = vertices[:, 0] - vertices[:, 2]
    # edges = np.array([e_0, e_1, e_2]).T
    # edges = get_shape_edges(vertices)
    return get_shape_edges(vertices)


def check_triangle_vertices_consistency(vertices: ndarray):
    """

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if number_of_vertices != 3:
        raise GeometryError("a triangle is defined by 3 points, not {}".format(number_of_vertices))
    if euclidean_dimension < 2:
        raise GeometryError("a triangle is defined in 2 or 3 dimension, not {}".format(euclidean_dimension))
    check_points_non_alignment(vertices)


def get_triangle_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edges = get_triangle_edges(vertices)
    # edges = get_shape_edges(vertices)
    number_of_edges = edges.shape[1]
    triangle_diameter = 0.0
    for i in range(number_of_edges):
        diam = np.linalg.norm(edges[:, i])
        if diam > triangle_diameter:
            triangle_diameter = diam
    # triangle_diameter = np.max([np.linalg.norm(e) for e in edges])
    # triangle_diameter = np.max([np.linalg.norm(edges[:, i]) for i in range(number_of_edges)])
    # triangle_diameter = max([get_euclidean_norm(e) for e in edges])
    return triangle_diameter


def get_triangle_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    edges = get_triangle_edges(vertices)
    # edges = get_shape_edges(vertices)
    if euclidean_dimension == 3:
        p = get_triangle_rotation_matrix(vertices)
    # e0 = (p @ edges.T).T
        e0 = p @ edges
    elif euclidean_dimension == 2:
        e0 = edges
    else:
        raise GeometryError("euclidean dimension must be either 2 or 3, not {}".format(euclidean_dimension))
    triangle_volume = np.abs(1.0 / 2.0 * np.linalg.det(e0[0:2, 0:2]))
    return triangle_volume


def get_triangle_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    triangle_centroid = get_triangle_barycenter(vertices)
    return triangle_centroid


def get_triangle_rotation_matrix(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 3:
        # e_0 = vertices[0] - vertices[-1]
        # e_0 = vertices[2] - vertices[0]
        e_0 = (vertices[:, 2] - vertices[:, 0])
        # e_0 = e_0 / np.linalg.norm(e_0)
        # e_t = vertices[1] - vertices[-1]
        # e_t = vertices[1] - vertices[0]
        e_t = (vertices[:, 1] - vertices[:, 0])
        # e_t = e_t / np.linalg.norm(e_t)
        e_2 = np.cross(e_0, e_t)
        e_1 = np.cross(e_2, e_0)
        e_0_n = e_0 / np.linalg.norm(e_0)
        e_1_n = e_1 / np.linalg.norm(e_1)
        e_2_n = e_2 / np.linalg.norm(e_2)
        # triangle_rotation_matrix = np.array([e_0, e_1, e_2])
        triangle_rotation_matrix = np.array([e_0_n, e_1_n, e_2_n])
    else:
        raise GeometryError("a triangle is a face only if the euclidean dimension is 3, not {}".format(euclidean_dimension))
    return triangle_rotation_matrix


def get_triangle_quadrature_size(integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS) -> int:
    """

    Args:
        integration_order:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_triangle.get_number_of_quadrature_points_in_triangle(integration_order)
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_size


def get_triangle_quadrature_weights(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 2:
        projected_vertices = vertices
    elif euclidean_dimension == 3:
        rotation_matrix = get_triangle_rotation_matrix(vertices)
        projected_vertices = (rotation_matrix @ vertices)[:-1]
    else:
        raise GeometryError("euclidean dimension must be either 2 or 3, not {}".format(euclidean_dimension))
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_triangle.get_number_of_quadrature_points_in_triangle(integration_order)
        quadrature_reference_weights = gauss_triangle.get_reference_triangle_quadrature_item(
            integration_order, QuadratureItem.WEIGHTS
        )
        jacobian_weights = np.zeros((quadrature_size,), dtype=real)
        jacobian_operators = gauss_triangle.get_reference_triangle_quadrature_item(
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
    quadrature_weights = quadrature_reference_weights * jacobian_weights
    return quadrature_weights


def get_triangle_quadrature_points(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_reference_points = gauss_triangle.get_reference_triangle_quadrature_item(
            integration_order, QuadratureItem.POINTS
        )
        quadrature_points = (quadrature_reference_points @ vertices.T).T
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_points
