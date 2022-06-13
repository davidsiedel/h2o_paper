from h2o.geometry.geometry import *
import h2o.geometry.shapes.shape_tetrahedron as shape_tetrahedron
from h2o.geometry.shapes.shape_polygon import get_polygon_partition


def get_polyhedron_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_barycenter(vertices)


def get_polyhedron_edges(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_edges(vertices)


def get_polyhedron_partition_number(vertices: ndarray, connectivity: List[List[int]]) -> int:
    """

    Args:
        vertices:
        connectivity:

    Returns:

    """
    partition_number = 0
    for faces_vertices_indices in connectivity:
        faces_vertices = vertices[:, faces_vertices_indices]
        number_of_vertices_in_face = faces_vertices.shape[1]
        if number_of_vertices_in_face == 3:
            partition_number += 1
        else:
            partition_number += number_of_vertices_in_face
    return partition_number


def get_polyhedron_partition(vertices: ndarray, connectivity: List[List[int]]) -> ndarray:
    """

    Args:
        vertices:
        connectivity:

    Returns:

    """
    polyhedron_barycenter = get_polyhedron_barycenter(vertices)
    polyhedron_partition_number = get_polyhedron_partition_number(vertices, connectivity)
    polyhedron_partition = np.zeros((polyhedron_partition_number, 3, 4), dtype=real)
    partition_count = 0
    for faces_vertices_indices in connectivity:
        faces_vertices = vertices[:, faces_vertices_indices]
        number_of_vertices_in_face = faces_vertices.shape[1]
        if number_of_vertices_in_face == 3:
            local_partition = np.zeros((3, 4), dtype=real)
            local_partition[:, :-1] = faces_vertices
            local_partition[:, -1] = polyhedron_barycenter
            polyhedron_partition[partition_count] = local_partition
            partition_count += 1
        else:
            polygon_partition = get_polygon_partition(faces_vertices)
            for i in range(number_of_vertices_in_face):
                local_partition = np.zeros((3, 4), dtype=real)
                local_partition[:, :-1] = polygon_partition[i]
                local_partition[:, -1] = polyhedron_barycenter
                polyhedron_partition[partition_count] = local_partition
                partition_count += 1
    return polyhedron_partition


def check_polyhedron_vertices_consistency(vertices: ndarray):
    """

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if number_of_vertices < 8:
        raise GeometryError("a tetrahedron is defined by 3 points, not {}".format(number_of_vertices))
    if euclidean_dimension < 3:
        raise GeometryError("a tetrahedron is defined in 3 dimension, not {}".format(euclidean_dimension))


def get_polyhedron_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    edges = get_polyhedron_edges(vertices)
    number_of_edges = edges.shape[1]
    polyhedron_diameter = 0.0
    for i in range(number_of_edges):
        diam = np.linalg.norm(edges[:, i])
        if diam > polyhedron_diameter:
            polyhedron_diameter = diam
    return polyhedron_diameter


def get_polyhedron_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    # euclidean_dimension = vertices.shape[0]
    # allo = np.zeros((3, 3), dtype=real)
    # allo[:, 0] = vertices[:, 1] - vertices[:, 0]
    # allo[:, 1] = vertices[:, 2] - vertices[:, 0]
    # allo[:, 2] = vertices[:, 3] - vertices[:, 0]
    # polyhedron_volume = np.abs(1.0 / 6.0 * np.linalg.det(allo))
    return 0.0


def get_polyhedron_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    polyhedron_centroid = get_polyhedron_barycenter(vertices)
    return polyhedron_centroid


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


def get_polyhedron_quadrature_size(
    vertices: ndarray,
    connectivity: List[List[int]],
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> int:
    """

    Args:
        vertices:
        connectivity:
        integration_order:
        quadrature_type:

    Returns:

    """
    tetrahedron_quadrature_size = shape_tetrahedron.get_tetrahedron_quadrature_size(
        integration_order, quadrature_type=quadrature_type
    )
    polyhedron_partition_number = get_polyhedron_partition_number(vertices, connectivity)
    quadrature_size = polyhedron_partition_number * tetrahedron_quadrature_size
    return quadrature_size


def get_polyhedron_quadrature_points(
    vertices: ndarray,
    connectivity: List[List[int]],
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        vertices:
        connectivity:
        integration_order:
        quadrature_type:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    quadrature_size = get_polyhedron_quadrature_size(
        vertices, connectivity, integration_order, quadrature_type=quadrature_type
    )
    quadrature_points = np.zeros((euclidean_dimension, quadrature_size), dtype=real)
    tetrahedrons = get_polyhedron_partition(vertices, connectivity)
    tetrahedron_quadrature_size = shape_tetrahedron.get_tetrahedron_quadrature_size(
        integration_order, quadrature_type=quadrature_type
    )
    for i, tetrahedron in enumerate(tetrahedrons):
        col0 = i * tetrahedron_quadrature_size
        col1 = (i + 1) * tetrahedron_quadrature_size
        tetrahedron_quadrature_points = shape_tetrahedron.get_tetrahedron_quadrature_points(
            tetrahedron, integration_order, quadrature_type=quadrature_type
        )
        quadrature_points[:, col0:col1] = tetrahedron_quadrature_points
    return quadrature_points


def get_polyhedron_quadrature_weights(
    vertices: ndarray,
    connectivity: List[List[int]],
    integration_order: int,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        vertices:
        connectivity:
        integration_order:
        quadrature_type:

    Returns:

    """
    polyhedron_quadrature_size = get_polyhedron_quadrature_size(
        vertices, connectivity, integration_order, quadrature_type=quadrature_type
    )
    quadrature_weights = np.zeros((polyhedron_quadrature_size,), dtype=real)
    tetrahedrons = get_polyhedron_partition(vertices, connectivity)
    tetrahedron_quadrature_size = shape_tetrahedron.get_tetrahedron_quadrature_size(
        integration_order, quadrature_type=quadrature_type
    )
    for i, tetrahedron in enumerate(tetrahedrons):
        col0 = i * tetrahedron_quadrature_size
        col1 = (i + 1) * tetrahedron_quadrature_size
        tetrahedron_quadrature_weights = shape_tetrahedron.get_tetrahedron_quadrature_weights(
            tetrahedron, integration_order, quadrature_type=quadrature_type
        )
        quadrature_weights[col0:col1] = tetrahedron_quadrature_weights
    return quadrature_weights
