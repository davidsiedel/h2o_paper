from typing import List, Dict

from numpy import ndarray

from h2o.mesh.element_description import CellDescription, FaceDescription
from h2o.h2o import *

geof_cell_library = {
    # --- 1D
    "c1d2": CellDescription(ShapeType.SEGMENT,
                            [
                                [0],
                                [1],
                            ],
                            2, 2, 1,
                            "c1d2"
                            ),
    # --- 2D
    "c2d3": CellDescription(ShapeType.TRIANGLE,
                            [
                                [0, 1],
                                [1, 2],
                                [2, 0],
                            ]
                            ,
                            3, 3, 2,
                            "c2d3"
                            ),
    "c2d4": CellDescription(ShapeType.QUADRANGLE,
                            [
                                [0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 0],
                            ],
                            4, 4, 2,
                            "c2d4"
                            ),
    "c2d5": CellDescription(ShapeType.POLYGON,
                            [
                                [0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 4],
                                [4, 0],
                            ],
                            5, 5, 2,
                            "c2d5"
                            ),
    # --- 3D
    "c3d8": CellDescription(ShapeType.HEXAHEDRON,
                            [
                                [0, 1, 2, 3],
                                [4, 7, 6, 5],
                                [1, 5, 6, 2],
                                [0, 3, 7, 4],
                                [3, 2, 6, 7],
                                [0, 4, 5, 1],
                            ],
                            8, 6, 3,
                            "c3d8"
                            ),
    "c3d9": CellDescription(ShapeType.POLYHEDRON,
                            [
                                [0, 1, 2, 3],
                                [1, 5, 7, 2],
                                [0, 3, 8, 4],
                                [3, 2, 7, 8],
                                [0, 4, 5, 1],
                                [6, 5, 4],
                                [6, 7, 5],
                                [6, 8, 7],
                                [6, 4, 8],
                            ],
                            9, 9, 3,
                            "c3d9"
                            ),
}
# [0, 1, 2, 3],
# [0, 4, 5, 1],
# [1, 5, 7, 2],
# [2, 7, 8, 3],
# [3, 8, 4, 0],
# [6, 5, 4],
# [6, 7, 5],
# [6, 8, 7],
# [6, 4, 8],

geof_face_library = {
    "c1d1": FaceDescription(ShapeType.POINT, 1, 1, "c1d1"),
    "c1d2": FaceDescription(ShapeType.SEGMENT, 2, 1, "c1d2"),
    "c2d3": FaceDescription(ShapeType.TRIANGLE, 3, 2, "c2d3"),
    "c2d4": FaceDescription(ShapeType.QUADRANGLE, 4, 2, "c2d4"),
    "c2d5": FaceDescription(ShapeType.POLYGON, 5, 2, "c2d5"),
}


def get_face_label(number_of_vertices: int) -> str:
    """

    Args:
        number_of_vertices: the number of vertices in the mesh

    Returns:

    """
    if number_of_vertices == 1:
        return "c1d1"
    elif number_of_vertices == 2:
        return "c1d2"
    else:
        return "c2d{}".format(number_of_vertices)


def get_mesh_file_content(geof_file_path: str) -> List[str]:
    """

    Args:
        geof_file_path: the file path to the mesh to read

    Returns:

    """
    with open(geof_file_path, "r") as geof_file:
        c = geof_file.readlines()
    return c


def get_vertices(geof_file_path: str) -> ndarray:
    """

    Args:
        geof_file_path: the file path to the mesh to read

    Returns:

    """
    c = get_mesh_file_content(geof_file_path)
    line_count = 2
    euclidean_dimension = int(c[line_count].rstrip().split(" ")[1])
    number_of_vertices_in_mesh = int(c[line_count].rstrip().split(" ")[0])
    line_count = 3
    vertices = np.zeros((euclidean_dimension, number_of_vertices_in_mesh), dtype=real)
    for vertex_count, i in enumerate(range(line_count, line_count + number_of_vertices_in_mesh)):
        vertex = [float(c[i].split(" ")[j]) for j in range(1, euclidean_dimension + 1)]
        vertex = np.array(vertex, dtype=real)
        vertices[:, vertex_count] += vertex
    return vertices


def get_vertices_boundaries_connectivity(geof_file_path: str) -> Dict[str, List[int]]:
    """

    Args:
        geof_file_path: the file path to the mesh to read

    Returns:

    """
    c = get_mesh_file_content(geof_file_path)
    line_count = 2
    number_of_vertices_in_mesh = int(c[line_count].rstrip().split(" ")[0])
    line_count = 4 + number_of_vertices_in_mesh
    number_of_cells_in_mesh = int(c[line_count].rstrip())
    global_line_count = 4 + number_of_vertices_in_mesh
    local_line_count = global_line_count + number_of_cells_in_mesh + 2
    vertices_boundaries_connectivity = {}
    for i in range(local_line_count, len(c)):
        if "**liset" in c[i] or "**elset" in c[i]:
            break
        else:
            if "**nset" in c[i]:
                boundary_name = (c[i]).split(" ")[1].rstrip()
                boundary_nodes = []
            else:
                if not "***" in c[i]:
                    if c[i].split(" ")[0] == "":
                        start_point = 1
                    else:
                        start_point = 0
                    nodes = [int(item.replace("\n", "")) - 1 for item in c[i].split(" ")[start_point:]]
                    boundary_nodes += nodes
                else:
                    vertices_boundaries_connectivity[boundary_name] = boundary_nodes
                    break
        vertices_boundaries_connectivity[boundary_name] = boundary_nodes
    return vertices_boundaries_connectivity


def get_cells_data(geof_file_path: str,) -> (List[List[int]], List[List[List[int]]], List[ShapeType], List[str]):
    """

    Args:
        geof_file_path: the file path to the mesh to read

    Returns:

    """
    c = get_mesh_file_content(geof_file_path)
    line_count = 2
    number_of_vertices_in_mesh = int(c[line_count].rstrip().split(" ")[0])
    line_count = 4 + number_of_vertices_in_mesh
    number_of_cells_in_mesh = int(c[line_count].rstrip())
    cells_vertices_connectivity = []
    cells_ordering = []
    cells_shapes = []
    cells_labels = []
    for cell_index in range(number_of_cells_in_mesh):
        line_count = 5 + number_of_vertices_in_mesh
        i = line_count + cell_index
        cell_label = str(c[i].split(" ")[1])
        cells_labels.append(cell_label)
        cell_vertices_connectivity = [int(c[i].split(" ")[j]) - 1 for j in range(2, len(c[i].split(" ")))]
        __check_vertices_connection_item(cell_vertices_connectivity, cell_label)
        cells_shapes.append(geof_cell_library[cell_label].shape_type)
        cells_vertices_connectivity.append(cell_vertices_connectivity)
        cells_ordering.append(geof_cell_library[cell_label].face_vertices_ordering)
    return cells_vertices_connectivity, cells_ordering, cells_shapes, cells_labels


def get_faces_data(
    cells_vertices_connectivity: List[List[int]], cells_label: List[str]
) -> (List[List[int]], List[List[int]]):
    """

    Args:
        cells_vertices_connectivity:
        cells_label:

    Returns:

    """
    faces_vertices_connectivity = []
    faces_shapes = []
    cells_faces_connectivity = []
    tags = []
    for v, l in zip(cells_vertices_connectivity, cells_label):
        cell_faces_connectivity = []
        g = geof_cell_library[l].face_vertices_ordering
        for u in g:
            c = [v[k] for k in u]
            tag = "".join([str(item).zfill(20) for item in np.sort(c)])
            if tag in tags:
                face_global_index = tags.index(tag)
                cell_faces_connectivity.append(face_global_index)
            else:
                tags.append(tag)
                face_label = get_face_label(len(c))
                __check_faces_connection_item(c, face_label)
                faces_vertices_connectivity.append(c)
                faces_shapes.append(geof_face_library[face_label].shape_type)
                face_global_index = len(faces_vertices_connectivity) - 1
                cell_faces_connectivity.append(face_global_index)
        cells_faces_connectivity.append(cell_faces_connectivity)
    return faces_vertices_connectivity, cells_faces_connectivity, faces_shapes


def get_vertices_weights(geof_file_path: str, items_vertices_connectivity: List[List[int]]):
    """

    Args:
        geof_file_path:
        items_vertices_connectivity:

    Returns:

    """
    c = get_mesh_file_content(geof_file_path)
    line_count = 2
    number_of_vertices_in_mesh = int(c[line_count].rstrip().split(" ")[0])
    vertices_weights_cell_wise = np.zeros((number_of_vertices_in_mesh,), dtype=size_type)
    for cell_vertices_connectivity in items_vertices_connectivity:
        vertices_weights_cell_wise[cell_vertices_connectivity] += np.ones(
            (len(cell_vertices_connectivity),), dtype=size_type
        )
    return vertices_weights_cell_wise


def get_faces_boundaries_connectivity(
    vertices_boundaries_connectivity: Dict[str, List[int]], faces_vertices_connectivity
):
    """

    Args:
        vertices_boundaries_connectivity:
        faces_vertices_connectivity:

    Returns:

    """
    faces_boundaries_connectivity = {}
    for key, val in vertices_boundaries_connectivity.items():
        faces_boundaries_connectivity[key] = []
        for face_index, face_vertices_connectivity in enumerate(faces_vertices_connectivity):
            res = True
            for vertex_index in face_vertices_connectivity:
                if not vertex_index in val:
                    res = False
            if res:
                faces_boundaries_connectivity[key] += [face_index]
    return faces_boundaries_connectivity


def __check_vertices_connection_item(val: List[int], cell_label: str):
    """

    Args:
        val:
        cell_label:

    Returns:

    """
    if isinstance(val, list):
        if not len(val) == geof_cell_library[cell_label].number_of_vertices:
            raise ValueError("wrong initiation")
        for item in val:
            if not isinstance(item, int):
                raise TypeError("item in val must be ints")
    else:
        raise TypeError("val must be a list of int")
    return


def __check_faces_connection_item(val: List[int], cell_label: str):
    """

    Args:
        val:
        cell_label:

    Returns:

    """
    if isinstance(val, list):
        if not len(val) == geof_cell_library[cell_label].number_of_faces:
            raise ValueError("wrong initiation")
        for item in val:
            if not isinstance(item, int):
                raise TypeError("item in val must be ints")
    else:
        raise TypeError("val must be a list of int")
    return
