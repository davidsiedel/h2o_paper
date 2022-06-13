import h2o.mesh.parsers.geof as geof
import h2o.mesh.gmsh.expprt as mshh
import h2o.geometry.shape as shp
from h2o.h2o import *


def get_number_of_quadrature_points_in_mesh(
    items_shapes: List[ShapeType],
    items_vertices: ndarray,
    integration_order: int,
    items_connectivity: List[List[List[int]]] = None,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> int:
    """

    Args:
        items_shapes:
        items_vertices:
        integration_order:
        items_connectivity:
        quadrature_type:

    Returns:

    """
    number_of_quadrature_points_in_mesh = 0
    for item_shape, item_vertices, item_connectivity in zip(items_shapes, items_vertices, items_connectivity):
        n = shp.get_quadrature_size(
            item_shape,
            item_vertices,
            integration_order,
            connectivity=item_connectivity,
            quadrature_type=quadrature_type,
        )
        number_of_quadrature_points_in_mesh += n
    return number_of_quadrature_points_in_mesh


class Mesh:
    vertices: ndarray
    euclidean_dimension: int
    cells_vertices_connectivity: List[List[int]]
    cells_ordering: List[List[List[int]]]
    cells_shape_types: List[ShapeType]
    faces_vertices_connectivity: List[List[int]]
    faces_shape_types: List[ShapeType]
    cells_faces_connectivity: List[List[int]]
    vertices_boundaries_connectivity: Dict[str, List[int]]
    faces_boundaries_connectivity: Dict[str, List[int]]
    number_of_vertices_in_mesh: int
    number_of_cells_in_mesh: int
    number_of_faces_in_mesh: int
    number_of_face_quadrature_points_in_mesh: int
    number_of_cell_quadrature_points_in_mesh: int
    vertices_weights_cell: ndarray
    vertices_weights_face: ndarray

    def __init__(
        self, mesh_file_path: str, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ):
        """

        Args:
            mesh_file_path: the file path to the mesh to read
            integration_order: the polynomial integration order, depending on the finite element order
            quadrature_type: the type of quadrature used to compute quadrature points and weights
        """
        if ".geof" in mesh_file_path:
            vertices = geof.get_vertices(mesh_file_path)
            (cells_vertices_connectivity, cells_ordering, cells_shapes, cells_labels) = geof.get_cells_data(
                mesh_file_path
            )
            (faces_vertices_connectivity, cells_faces_connectivity, faces_shapes) = geof.get_faces_data(
                cells_vertices_connectivity, cells_labels
            )
            vertices_boundaries_connectivity = geof.get_vertices_boundaries_connectivity(mesh_file_path)
            faces_boundaries_connectivity = geof.get_faces_boundaries_connectivity(
                vertices_boundaries_connectivity, faces_vertices_connectivity
            )
            self.vertices = vertices
            self.euclidean_dimension = self.vertices.shape[0]
            self.cells_vertices_connectivity = cells_vertices_connectivity
            self.cells_ordering = cells_ordering
            self.cells_shape_types = cells_shapes
            self.faces_vertices_connectivity = faces_vertices_connectivity
            self.faces_shape_types = faces_shapes
            self.cells_faces_connectivity = cells_faces_connectivity
            self.vertices_boundaries_connectivity = vertices_boundaries_connectivity
            self.faces_boundaries_connectivity = faces_boundaries_connectivity
            self.number_of_vertices_in_mesh = self.get_number_of_vertices_in_mesh()
            self.number_of_cells_in_mesh = self.get_number_of_cells_in_mesh()
            self.number_of_faces_in_mesh = self.get_number_of_faces_in_mesh()
            # self.number_of_cell_quadrature_points_in_mesh = get_number_of_quadrature_points_in_mesh(
            #     cells_shapes, integration_order, quadrature_type=quadrature_type
            # )
            self.number_of_cell_quadrature_points_in_mesh = self.get_number_of_cell_quadrature_points_in_mesh(
                integration_order, quadrature_type=quadrature_type
            )
            self.number_of_face_quadrature_points_in_mesh = self.get_number_of_face_quadrature_points_in_mesh(
                integration_order, quadrature_type=quadrature_type
            )
            # n_fq = 0
            # for _f in range(self.number_of_faces_in_mesh):
            #     face_vertices = self.vertices[:, self.faces_vertices_connectivity[_f]]
            #     face_shape_type = self.faces_shape_types[_f]
            #     n = shp.get_quadrature_size(
            #         face_shape_type, face_vertices, integration_order, quadrature_type=quadrature_type,
            #     )
            #     n_fq += n
            # self.number_of_face_quadrature_points_in_mesh = get_number_of_quadrature_points_in_mesh(
            #     faces_shapes, integration_order, quadrature_type=quadrature_type
            # )
            self.vertices_weights_cell = geof.get_vertices_weights(mesh_file_path, cells_vertices_connectivity)
            self.vertices_weights_face = geof.get_vertices_weights(mesh_file_path, faces_vertices_connectivity)
        elif ".msh" in mesh_file_path:
            (
                vertices,
                euclidean_dimension,
                cells_vertices_connectivity,
                cells_ordering,
                cells_shape_types,
                faces_vertices_connectivity,
                faces_shape_types,
                cells_faces_connectivity,
                vertices_boundaries_connectivity,
                faces_boundaries_connectivity,
                number_of_vertices_in_mesh,
                number_of_cells_in_mesh,
                number_of_faces_in_mesh,
                vertices_weights_cell_wise,
                vertices_weights_face_wise,
            ) = mshh.build_mesh(mesh_file_path)
            self.vertices = vertices
            self.euclidean_dimension = euclidean_dimension
            self.cells_vertices_connectivity = cells_vertices_connectivity
            self.cells_ordering = cells_ordering
            self.cells_shape_types = cells_shape_types
            self.faces_vertices_connectivity = faces_vertices_connectivity
            self.faces_shape_types = faces_shape_types
            self.cells_faces_connectivity = cells_faces_connectivity
            self.vertices_boundaries_connectivity = vertices_boundaries_connectivity
            self.faces_boundaries_connectivity = faces_boundaries_connectivity
            self.number_of_vertices_in_mesh = number_of_vertices_in_mesh
            self.number_of_cells_in_mesh = number_of_cells_in_mesh
            self.number_of_faces_in_mesh = number_of_faces_in_mesh
            self.vertices_weights_cell = vertices_weights_cell_wise
            self.vertices_weights_face = vertices_weights_face_wise
            self.number_of_cell_quadrature_points_in_mesh = self.get_number_of_cell_quadrature_points_in_mesh(
                integration_order, quadrature_type=quadrature_type
            )
            self.number_of_face_quadrature_points_in_mesh = self.get_number_of_face_quadrature_points_in_mesh(
                integration_order, quadrature_type=quadrature_type
            )
            # print(self.faces_boundaries_connectivity)
        else:
            raise IOError("unsupported mesh file format")

    def get_number_of_cells_in_mesh(self) -> int:
        """

        Returns: the number of cells in the mesh

        """
        number_of_cells_in_mesh = len(self.cells_vertices_connectivity)
        return number_of_cells_in_mesh

    def get_number_of_faces_in_mesh(self) -> int:
        """

        Returns: the number of faces in the mesh

        """
        number_of_faces_in_mesh = len(self.faces_vertices_connectivity)
        return number_of_faces_in_mesh

    def get_number_of_vertices_in_mesh(self) -> int:
        """

        Returns: the number of vertices in the mesh

        """
        number_of_vertices_in_mesh = self.vertices.shape[1]
        return number_of_vertices_in_mesh

    def get_number_of_face_quadrature_points_in_mesh(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> int:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        n_fq = 0
        for _f in range(self.number_of_faces_in_mesh):
            face_vertices = self.vertices[:, self.faces_vertices_connectivity[_f]]
            face_shape_type = self.faces_shape_types[_f]
            n = shp.get_quadrature_size(
                face_shape_type,
                face_vertices,
                integration_order,
                quadrature_type=quadrature_type,
            )
            n_fq += n
        return n_fq

    def get_number_of_cell_quadrature_points_in_mesh(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> int:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        n_cq = 0
        for _c in range(self.number_of_cells_in_mesh):
            cell_vertices = self.vertices[:, self.cells_vertices_connectivity[_c]]
            cell_shape_type = self.cells_shape_types[_c]
            cell_connectivity = self.cells_ordering[_c]
            n = shp.get_quadrature_size(
                cell_shape_type,
                cell_vertices,
                integration_order,
                connectivity=cell_connectivity,
                quadrature_type=quadrature_type,
            )
            n_cq += n
        return n_cq
