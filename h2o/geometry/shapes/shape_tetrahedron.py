from h2o.geometry.geometry import *
import h2o.geometry.quadratures.gauss.gauss_tetrahedron as gauss_tetrahedron
from h2o.geometry.quadratures.quadrature import *


def get_tetrahedron_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_barycenter(vertices)


def get_tetrahedron_edges(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_edges(vertices)


def check_tetrahedron_vertices_consistency(vertices: ndarray):
    """

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if number_of_vertices != 4:
        raise GeometryError("a tetrahedron is defined by 3 points, not {}".format(number_of_vertices))
    if euclidean_dimension < 3:
        raise GeometryError("a tetrahedron is defined in 3 dimension, not {}".format(euclidean_dimension))


def get_tetrahedron_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edges = get_tetrahedron_edges(vertices)
    number_of_edges = edges.shape[1]
    tetrahedron_diameter = 0.0
    for i in range(number_of_edges):
        diam = np.linalg.norm(edges[:, i])
        if diam > tetrahedron_diameter:
            tetrahedron_diameter = diam
    return tetrahedron_diameter


def get_tetrahedron_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    allo = np.zeros((3, 3), dtype=real)
    allo[:, 0] = vertices[:, 1] - vertices[:, 0]
    allo[:, 1] = vertices[:, 2] - vertices[:, 0]
    allo[:, 2] = vertices[:, 3] - vertices[:, 0]
    tetrahedron_volume = np.abs(1.0 / 6.0 * np.linalg.det(allo))
    return tetrahedron_volume


def get_tetrahedron_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    tetrahedron_centroid = get_tetrahedron_barycenter(vertices)
    return tetrahedron_centroid


# def get_tetrahedron_rotation_matrix(vertices: ndarray) -> ndarray:
#     """
#
#     Args:
#         vertices:
#
#     Returns:
#
#     """
#     euclidean_dimension = vertices.shape[0]
#     if euclidean_dimension == 3:
#         # e_0 = vertices[0] - vertices[-1]
#         # e_0 = vertices[2] - vertices[0]
#         e_0 = vertices[:, 2] - vertices[:, 0]
#         e_0 = e_0 / np.linalg.norm(e_0)
#         # e_t = vertices[1] - vertices[-1]
#         # e_t = vertices[1] - vertices[0]
#         e_t = vertices[:, 1] - vertices[:, 0]
#         e_t = e_t / np.linalg.norm(e_t)
#         e_2 = np.cross(e_0, e_t)
#         e_1 = np.cross(e_2, e_0)
#         tetrahedron_rotation_matrix = np.array([e_0, e_1, e_2])
#     else:
#         raise GeometryError("a tetrahedron is a face only if the euclidean dimension is 3, not {}".format(euclidean_dimension))
#     return tetrahedron_rotation_matrix


def get_tetrahedron_quadrature_size(integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS) -> int:
    """

    Args:
        integration_order:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_tetrahedron.get_number_of_quadrature_points_in_tetrahedron(integration_order)
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_size


def get_tetrahedron_quadrature_weights(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_tetrahedron.get_number_of_quadrature_points_in_tetrahedron(integration_order)
        quadrature_reference_weights = gauss_tetrahedron.get_reference_tetrahedron_quadrature_item(
            integration_order, QuadratureItem.WEIGHTS
        )
        jacobian_weights = np.zeros((quadrature_size,), dtype=real)
        jacobian_operators = gauss_tetrahedron.get_reference_tetrahedron_quadrature_item(
            integration_order, QuadratureItem.JACOBIAN
        )
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    for i, jacobian_operator in enumerate(jacobian_operators):
        jacobian = np.zeros((3, 3), dtype=real)
        jacobian[0, 0] = jacobian_operator[0] @ vertices[0, :]
        jacobian[0, 1] = jacobian_operator[1] @ vertices[0, :]
        jacobian[0, 2] = jacobian_operator[2] @ vertices[0, :]
        jacobian[1, 0] = jacobian_operator[3] @ vertices[1, :]
        jacobian[1, 1] = jacobian_operator[4] @ vertices[1, :]
        jacobian[1, 2] = jacobian_operator[5] @ vertices[1, :]
        jacobian[2, 0] = jacobian_operator[6] @ vertices[2, :]
        jacobian[2, 1] = jacobian_operator[7] @ vertices[2, :]
        jacobian[2, 2] = jacobian_operator[8] @ vertices[2, :]
        jacobian_weights[i] = np.abs(np.linalg.det(jacobian))
    quadrature_weights = quadrature_reference_weights * jacobian_weights
    return quadrature_weights


def get_tetrahedron_quadrature_points(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_reference_points = gauss_tetrahedron.get_reference_tetrahedron_quadrature_item(
            integration_order, QuadratureItem.POINTS
        )
        quadrature_points = (quadrature_reference_points @ vertices.T).T
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_points
