from h2o.geometry.geometry import *
import h2o.geometry.quadratures.gauss.gauss_segment as gauss_segment
from h2o.geometry.quadratures.quadrature import *


def get_segment_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_barycenter(vertices)


def check_segment_vertices_consistency(vertices: ndarray):
    """

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    if number_of_vertices != 2:
        raise GeometryError("a segment is defined by 2 points, not {}".format(number_of_vertices))
    distance = get_segment_volume(vertices)
    if distance == 0:
        raise GeometryError("points are the same")


def get_segment_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    diameter = get_segment_volume(vertices)
    return diameter


def get_segment_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edge = vertices[:, 0] - vertices[:, 1]
    volume = np.linalg.norm(edge)
    return volume


def get_segment_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    centroid = get_segment_barycenter(vertices)
    return centroid


def get_segment_rotation_matrix(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns: the rotation matrix of the segment if it is a face in a 2-dimensional euclidean space, or the identity
    matrix otherwise

    """
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 1:
        mapping_matrix = np.eye(1)
    elif euclidean_dimension == 2:
        e_0 = vertices[:, 1] - vertices[:, 0]
        e_0 = e_0 / np.linalg.norm(e_0)
        e_1 = np.array([e_0[1], -e_0[0]])
        mapping_matrix = np.array([e_0, e_1])
    else:
        raise GeometryError("euclidean dimension must be either 1 or 2, not {}".format(euclidean_dimension))
    return mapping_matrix


def get_segment_quadrature_size(integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS) -> int:
    """

    Args:
        integration_order:
        quadrature_type:

    Returns:

    """
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_segment.get_number_of_quadrature_points_in_segment(integration_order)
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_size


def get_segment_quadrature_weights(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 1:
        projected_vertices = vertices
    elif euclidean_dimension == 2:
        rotation_matrix = get_segment_rotation_matrix(vertices)
        projected_vertices = (rotation_matrix @ vertices)[:-1]
    else:
        raise GeometryError("euclidean dimension must be either 2 or 3, not {}".format(euclidean_dimension))
    if quadrature_type == QuadratureType.GAUSS:
        quadrature_size = gauss_segment.get_number_of_quadrature_points_in_segment(integration_order)
        quadrature_reference_weights = gauss_segment.get_reference_segment_quadrature_item(
            integration_order, QuadratureItem.WEIGHTS
        )
        jacobian_weights = np.zeros((quadrature_size,), dtype=real)
        jacobian_operators = gauss_segment.get_reference_segment_quadrature_item(
            integration_order, QuadratureItem.JACOBIAN
        )
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    for i, jaco in enumerate(jacobian_operators):
        jacobian_weights[i] = np.abs(jaco[0] @ projected_vertices[0, :])
    quadrature_weights = quadrature_reference_weights * jacobian_weights
    return quadrature_weights


def get_segment_quadrature_points(
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
        quadrature_reference_points = gauss_segment.get_reference_segment_quadrature_item(
            integration_order, QuadratureItem.POINTS
        )
        quadrature_points = (quadrature_reference_points @ vertices.T).T
    else:
        raise QuadratureError("no such quadrature type as {}".format(quadrature_type))
    return quadrature_points
