from h2o.h2o import *
from scipy.special import binom


def get_shape_barycenter(vertices: ndarray) -> ndarray:
    """
    Find the barycenter of a shape

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    number_of_vertices = vertices.shape[1]
    shape_barycenter = np.zeros((euclidean_dimension,), dtype=real)
    for i in range(number_of_vertices):
        shape_barycenter += vertices[:, i]
    shape_barycenter *= (1.0 / number_of_vertices)
    return shape_barycenter


def get_shape_bounding_box(vertices: ndarray) -> ndarray:
    """
    Find the bounding box of a shape

    Args:
        vertices:

    Returns:

    """
    euclidean_dimension = vertices.shape[0]
    number_of_vertices = vertices.shape[1]
    bounding_box = np.zeros((euclidean_dimension,), dtype=real)
    for i in range(number_of_vertices):
        for j in range(number_of_vertices):
            if j > i:
                for k in range(euclidean_dimension):
                    if np.abs(vertices[k, i] - vertices[k, j]) > np.abs(bounding_box[k]):
                        bounding_box[k] = np.abs(vertices[k, i] - vertices[k, j])
                        # bounding_box[k] = vertices[k, i] - vertices[k, j]
    return bounding_box


def get_shape_edges(vertices: ndarray) -> ndarray:
    """
    Assuming the shape is a bone or a face, returns its edges

    Args:
        vertices:

    Returns:

    """
    number_of_vertices = vertices.shape[1]
    euclidean_dimension = vertices.shape[0]
    if number_of_vertices < 2:
        raise GeometryError("NO")
    elif number_of_vertices == 2:
        edges = vertices[:, 1] - vertices[:, 0]
        return edges
    else:
        if euclidean_dimension == 1:
            raise GeometryError("an edge with more that 2 vertices is forbidden in 1 dimension")
        elif euclidean_dimension == 2:
            edges = np.zeros((euclidean_dimension, number_of_vertices), dtype=real)
            for i in range(number_of_vertices):
                edges[:, i] = vertices[:, i] - vertices[:, i - 1]
            return edges
        elif euclidean_dimension == 3:
            num_comb = int(binom(number_of_vertices, 2))
            edges = np.zeros((euclidean_dimension, num_comb), dtype=real)
            edge_count = 0
            for i in range(number_of_vertices):
                for j in range(number_of_vertices):
                    if i > j:
                        edges[:, edge_count] = vertices[:, i] - vertices[:, j]
                        edge_count += 1
            return edges
        else:
            raise GeometryError("euclidean dimesino must be either 1, 2, or 3, not {}".format(euclidean_dimension))


def check_points_non_alignment(vertices: ndarray):
    """
    Assuming the shape is a bone or a face, raises an error if points are aligned

    Args:
        vertices:
    """
    number_of_vertices = vertices.shape[1]
    if number_of_vertices < 3:
        raise GeometryError("NO")
    else:
        edges = get_shape_edges(vertices)
        number_of_edges = edges.shape[1]
        for i in range(number_of_edges - 1):
            edge0 = edges[:, i]/np.linalg.norm(edges[:, i])
            edge1 = edges[:, i + 1]/np.linalg.norm(edges[:, i + 1])
            cos_theta = edge0 @ edge1
            tol = 1.e-10
            if (0.0 - tol > cos_theta > 0.0 + tol) or (-(1.0 - tol) > cos_theta > -(1.0 + tol)):
                raise GeometryError("points are aligned")
                # pass


def check_points_coplanar(vertices: ndarray):
    """
    Assuming the shape is a face, raises an error if points are not coplanar

    Args:
        vertices:
    """
    euclidean_dimension = vertices.shape[0]
    number_of_vertices = vertices.shape[1]
    if euclidean_dimension != 3:
        raise GeometryError("coplanarity has no meaning in 2D or 1D")
    else:
        if number_of_vertices < 4:
            raise GeometryError("NO")
        else:
            edges = get_shape_edges(vertices)
            number_of_edges = edges.shape[1]
            edge0 = edges[:, 0] / np.linalg.norm(edges[:, 0])
            edge1 = edges[:, 1] / np.linalg.norm(edges[:, 1])
            normal_vector = np.cross(edge0, edge1)
            tol = 1.e-10
            for i in range(number_of_edges - 1):
                edge = edges[:, i] / np.linalg.norm(edges[:, i])
                cos_theta = edge @ normal_vector
                # if cos_theta < 0.0 + tol or cos_theta > 0.0 - tol or cos_theta < 1.0 + tol or cos_theta > 1.0 - tol:
                #     raise GeometryError("points are not coplanar")
                if not 0.0 - tol < cos_theta < 0.0 + tol and not -(1.0 - tol) < cos_theta < -(1.0 + tol):
                    raise GeometryError("points are not coplanar")
                    # pass
