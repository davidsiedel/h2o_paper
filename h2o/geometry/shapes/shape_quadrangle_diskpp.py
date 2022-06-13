from h2o.geometry.geometry import *
import h2o.geometry.shapes.shape_triangle as shape_triangle


def get_polygon_barycenter(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_barycenter(vertices)


def get_polygon_edges(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    return get_shape_edges(vertices)


def get_polygon_partition(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    polygon_partition = np.zeros((2, euclidean_dimension, 3), dtype=real)
    polygon_partition[0] = np.array([vertices[:, 0], vertices[:, 1], vertices[:, 2]], dtype=real).T
    polygon_partition[1] = np.array([vertices[:, 2], vertices[:, 3], vertices[:, 0]], dtype=real).T
    # polygon_partition[0] = np.array([vertices[:, 0], vertices[:, 1], vertices[:, 3]], dtype=real).T
    # polygon_partition[1] = np.array([vertices[:, 1], vertices[:, 2], vertices[:, 3]], dtype=real).T
    return polygon_partition


def check_polygon_vertices_consistency(vertices: ndarray):
    """

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if number_of_vertices < 5:
        raise GeometryError("a polygon is defined by more than 4 points, not {}".format(number_of_vertices))
    triangles = get_polygon_partition(vertices)
    for triangle_vertices in triangles:
        # shape_triangle.check_triangle_vertices_consistency(triangle_vertices)
        check_points_non_alignment(triangle_vertices)
    if euclidean_dimension == 3:
        check_points_coplanar(vertices)


def get_polygon_diameter(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    polygon_diameter = 0.0
    for i in range(number_of_vertices):
        v0 = vertices[:, i]
        for j in range(number_of_vertices):
            v1 = vertices[:, j]
            if not i == j:
                edge = v1 - v0
                edge_length = np.linalg.norm(edge)
                if edge_length > polygon_diameter:
                    polygon_diameter = edge_length
    return polygon_diameter


def get_lace(vertices: ndarray, index: int) -> float:
    """

    Args:
        vertices:
        index:

    Returns:

    """
    lace = vertices[0, index - 1] * vertices[1, index] - vertices[0, index] * vertices[1, index - 1]
    return lace


def get_polygon_signed_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    lace_sum = 0.0
    for i in range(number_of_vertices):
        lace = get_lace(vertices, i)
        lace_sum += lace
    polygon_volume = 1.0 / 2.0 * lace_sum
    return polygon_volume


def get_polygon_volume(vertices: ndarray) -> float:
    """

    Args:
        vertices:

    Returns:

    """
    # number_of_vertices = vertices.shape[1]
    # lace_sum = 0.0
    # for i in range(number_of_vertices):
    #     lace = get_lace(vertices, i)
    #     lace_sum += lace
    # polygon_volume = np.abs(1.0 / 2.0 * lace_sum)
    return np.abs(get_polygon_signed_volume(vertices))


def get_polygon_centroid(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if euclidean_dimension == 2:
        vprim = vertices
    elif euclidean_dimension == 3:
        rot = get_polygon_rotation_matrix(vertices)
        vprim = (rot @ vertices)[:-1, :]
    else:
        raise GeometryError("no")
    polygon_signed_volume = get_polygon_signed_volume(vprim)
    cx_sum = 0.0
    for i in range(number_of_vertices):
        cx_sum += (vprim[0, i - 1] + vprim[0, i]) * get_lace(vprim, i)
    polygon_centroid_x = 1.0 / (6.0 * polygon_signed_volume) * cx_sum
    cy_sum = 0.0
    for i in range(number_of_vertices):
        cy_sum += (vprim[1, i - 1] + vprim[1, i]) * get_lace(vprim, i)
    polygon_centroid_y = 1.0 / (6.0 * polygon_signed_volume) * cy_sum
    polygon_centroid = np.array([polygon_centroid_x, polygon_centroid_y])
    if euclidean_dimension == 3:
        polycent = np.zeros((euclidean_dimension,), dtype=real)
        polycent[:2] = polygon_centroid
        rot = get_polygon_rotation_matrix(vertices)
        h_t = (rot @ vertices)
        h = h_t[2, 0]
        polycent[2] = h
        polycent_fin = np.linalg.inv(rot) @ polycent
        return polycent_fin
    else:
        return polygon_centroid
    # return polygon_centroid


def get_polygon_rotation_matrix(vertices: ndarray) -> ndarray:
    """

    Args:
        vertices:

    Returns:

    """
    tri = vertices[:, :-1]
    p = shape_triangle.get_triangle_rotation_matrix(tri)
    return p


def get_polygon_quadrature_size(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> int:
    """

    Args:
        vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    triangle_quadrature_size = shape_triangle.get_triangle_quadrature_size(
        integration_order, quadrature_type=quadrature_type
    )
    polygon_quadrature_size = 2 * triangle_quadrature_size
    return polygon_quadrature_size


def get_polygon_quadrature_points(
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
    quadrature_size = get_polygon_quadrature_size(vertices, integration_order)
    quadrature_points = np.zeros((euclidean_dimension, quadrature_size), dtype=real)
    triangles = get_polygon_partition(vertices)
    triangle_quadrature_size = shape_triangle.get_triangle_quadrature_size(
        integration_order, quadrature_type=quadrature_type
    )
    for i, triangle in enumerate(triangles):
        col0 = i * triangle_quadrature_size
        col1 = (i + 1) * triangle_quadrature_size
        triangle_quadrature_points = shape_triangle.get_triangle_quadrature_points(
            triangle, integration_order, quadrature_type=quadrature_type
        )
        quadrature_points[:, col0:col1] = triangle_quadrature_points
    return quadrature_points


def get_polygon_quadrature_weights(
    vertices: ndarray, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
) -> ndarray:
    """

    Args:
        vertices:
        integration_order:
        quadrature_type:

    Returns:

    """
    polygon_quadrature_size = get_polygon_quadrature_size(vertices, integration_order)
    quadrature_weights = np.zeros((polygon_quadrature_size,), dtype=real)
    triangles = get_polygon_partition(vertices)
    triangle_quadrature_size = shape_triangle.get_triangle_quadrature_size(
        integration_order, quadrature_type=quadrature_type
    )
    for i, triangle in enumerate(triangles):
        col0 = i * triangle_quadrature_size
        col1 = (i + 1) * triangle_quadrature_size
        triangle_quadrature_weights = shape_triangle.get_triangle_quadrature_weights(
            triangle, integration_order, quadrature_type=quadrature_type
        )
        quadrature_weights[col0:col1] = triangle_quadrature_weights
    return quadrature_weights
