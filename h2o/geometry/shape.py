import h2o.geometry.shapes.shape_segment as isegment
import h2o.geometry.shapes.shape_triangle as itriangle
import h2o.geometry.shapes.shape_quadrangle as iquadrangle
import h2o.geometry.shapes.shape_quadrangle_diskpp as iquadrangle_diskpp
import h2o.geometry.shapes.shape_polygon as ipolygon
import h2o.geometry.shapes.shape_tetrahedron as itetrahedron
import h2o.geometry.shapes.shape_hexahedron as ihexahedron
import h2o.geometry.shapes.shape_polyhedron as ipolyhedron
from h2o.geometry.geometry import get_shape_bounding_box
from h2o.h2o import *

class QuadrangleImplementation(Enum):
    DISKPP = auto()
    NORMAL = auto()

WHICH_QUADRANGLE: QuadrangleImplementation = QuadrangleImplementation.NORMAL

def _check_shape(shape_type: ShapeType, shape_vertices: ndarray):
    """

    Args:
        shape_type:
        shape_vertices:
    """
    if shape_type == ShapeType.SEGMENT:
        isegment.check_segment_vertices_consistency(shape_vertices)
    elif shape_type == ShapeType.TRIANGLE:
        itriangle.check_triangle_vertices_consistency(shape_vertices)
    elif shape_type == ShapeType.QUADRANGLE:
        iquadrangle.check_quadrangle_vertices_consistency(shape_vertices)
    elif shape_type == ShapeType.POLYGON:
        ipolygon.check_polygon_vertices_consistency(shape_vertices)
    elif shape_type == ShapeType.TETRAHEDRON:
        itetrahedron.check_tetrahedron_vertices_consistency(shape_vertices)
    elif shape_type == ShapeType.HEXAHEDRON:
        ihexahedron.check_hexahedron_vertices_consistency(shape_vertices)
    elif shape_type == ShapeType.POLYHEDRON:
        ipolyhedron.check_polyhedron_vertices_consistency(shape_vertices)
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


def get_centroid(shape_type: ShapeType, vertices: ndarray) -> ndarray:
    """

    Args:
        shape_type:
        vertices:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_centroid(vertices)
    elif shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_centroid(vertices)
    elif shape_type == ShapeType.QUADRANGLE:
        return iquadrangle.get_quadrangle_centroid(vertices)
    elif shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_centroid(vertices)
    elif shape_type == ShapeType.TETRAHEDRON:
        return itetrahedron.get_tetrahedron_centroid(vertices)
    elif shape_type == ShapeType.HEXAHEDRON:
        return ihexahedron.get_hexahedron_centroid(vertices)
    elif shape_type == ShapeType.POLYHEDRON:
        return ipolyhedron.get_polyhedron_centroid(vertices)
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


def get_diameter(shape_type: ShapeType, vertices: ndarray) -> float:
    """

    Args:
        shape_type:
        vertices:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_diameter(vertices)
    elif shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_diameter(vertices)
    elif shape_type == ShapeType.QUADRANGLE:
        return iquadrangle.get_quadrangle_diameter(vertices)
    elif shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_diameter(vertices)
    elif shape_type == ShapeType.TETRAHEDRON:
        return itetrahedron.get_tetrahedron_diameter(vertices)
    elif shape_type == ShapeType.HEXAHEDRON:
        return ihexahedron.get_hexahedron_diameter(vertices)
    elif shape_type == ShapeType.POLYHEDRON:
        return ipolyhedron.get_polyhedron_diameter(vertices)
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


def get_volume(shape_type: ShapeType, vertices: ndarray) -> float:
    """

    Args:
        shape_type:
        vertices:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_volume(vertices)
    elif shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_volume(vertices)
    elif shape_type == ShapeType.QUADRANGLE:
        return iquadrangle.get_quadrangle_volume(vertices)
    elif shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_volume(vertices)
    elif shape_type == ShapeType.TETRAHEDRON:
        return itetrahedron.get_tetrahedron_volume(vertices)
    elif shape_type == ShapeType.HEXAHEDRON:
        return ihexahedron.get_hexahedron_volume(vertices)
    elif shape_type == ShapeType.POLYHEDRON:
        return ipolyhedron.get_polyhedron_volume(vertices)
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


def get_rotation_matrix(face_shape_type: ShapeType, face_vertices: ndarray) -> ndarray:
    """
    Args:
        face_shape_type:
        face_vertices:
    Returns:
    """
    if face_shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_rotation_matrix(face_vertices)
    elif face_shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_rotation_matrix(face_vertices)
    elif face_shape_type == ShapeType.QUADRANGLE:
        return iquadrangle.get_quadrangle_rotation_matrix(face_vertices)
    elif face_shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_rotation_matrix(face_vertices)
    elif face_shape_type == ShapeType.TETRAHEDRON:
        raise GeometryError("no rotation matrix in 3 dimension")
    elif face_shape_type == ShapeType.HEXAHEDRON:
        raise GeometryError("no rotation matrix in 3 dimension")
    elif face_shape_type == ShapeType.POLYHEDRON:
        raise GeometryError("no rotation matrix in 3 dimension")
    else:
        raise GeometryError("no such shape as {}".format(face_shape_type.value))


def get_quadrature_size(
    shape_type: ShapeType,
    vertices: ndarray,
    integration_order: int,
    connectivity: List[List[int]] = None,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> int:
    """

    Args:
        vertices:
        shape_type:
        integration_order:
        connectivity:
        quadrature_type:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_quadrature_size(integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_quadrature_size(integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.QUADRANGLE:
        if WHICH_QUADRANGLE == QuadrangleImplementation.DISKPP:
            return iquadrangle_diskpp.get_polygon_quadrature_size(vertices, integration_order,
                                                                  quadrature_type=quadrature_type)
        elif WHICH_QUADRANGLE == QuadrangleImplementation.NORMAL:
            return iquadrangle.get_quadrangle_quadrature_size(integration_order, quadrature_type=quadrature_type)
        else:
            raise KeyError
        # return ipolygon.get_polygon_quadrature_size(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_quadrature_size(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.TETRAHEDRON:
        return itetrahedron.get_tetrahedron_quadrature_size(integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.HEXAHEDRON:
        return ihexahedron.get_hexahedron_quadrature_size(integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.POLYHEDRON:
        return ipolyhedron.get_polyhedron_quadrature_size(
            vertices, connectivity, integration_order, quadrature_type=quadrature_type
        )
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


def get_quadrature_points(
    shape_type: ShapeType,
    vertices: ndarray,
    integration_order: int,
    connectivity: List[List[int]] = None,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        vertices:
        shape_type:
        integration_order:
        connectivity:
        quadrature_type:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_quadrature_points(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_quadrature_points(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.QUADRANGLE:
        if WHICH_QUADRANGLE == QuadrangleImplementation.DISKPP:
            return iquadrangle_diskpp.get_polygon_quadrature_points(
                vertices, integration_order, quadrature_type=quadrature_type
            )
        elif WHICH_QUADRANGLE == QuadrangleImplementation.NORMAL:
            return iquadrangle.get_quadrangle_quadrature_points(
                vertices, integration_order, quadrature_type=quadrature_type
            )
        else:
            raise KeyError
        # return ipolygon.get_polygon_quadrature_points(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_quadrature_points(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.TETRAHEDRON:
        return itetrahedron.get_tetrahedron_quadrature_points(
            vertices, integration_order, quadrature_type=quadrature_type
        )
    elif shape_type == ShapeType.HEXAHEDRON:
        return ihexahedron.get_hexahedron_quadrature_points(
            vertices, integration_order, quadrature_type=quadrature_type
        )
    elif shape_type == ShapeType.POLYHEDRON:
        return ipolyhedron.get_polyhedron_quadrature_points(
            vertices, connectivity, integration_order, quadrature_type=quadrature_type
        )
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


def get_quadrature_weights(
    shape_type: ShapeType,
    vertices: ndarray,
    integration_order: int,
    connectivity: List[List[int]] = None,
    quadrature_type: QuadratureType = QuadratureType.GAUSS,
) -> ndarray:
    """

    Args:
        vertices:
        shape_type:
        integration_order:
        connectivity:
        quadrature_type:

    Returns:

    """
    if shape_type == ShapeType.SEGMENT:
        return isegment.get_segment_quadrature_weights(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.TRIANGLE:
        return itriangle.get_triangle_quadrature_weights(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.QUADRANGLE:
        if WHICH_QUADRANGLE == QuadrangleImplementation.DISKPP:
            return iquadrangle_diskpp.get_polygon_quadrature_weights(
                vertices, integration_order, quadrature_type=quadrature_type
            )
        elif WHICH_QUADRANGLE == QuadrangleImplementation.NORMAL:
            return iquadrangle.get_quadrangle_quadrature_weights(
                vertices, integration_order, quadrature_type=quadrature_type
            )
        else:
            raise KeyError
        # return ipolygon.get_polygon_quadrature_weights(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.POLYGON:
        return ipolygon.get_polygon_quadrature_weights(vertices, integration_order, quadrature_type=quadrature_type)
    elif shape_type == ShapeType.TETRAHEDRON:
        return itetrahedron.get_tetrahedron_quadrature_weights(
            vertices, integration_order, quadrature_type=quadrature_type
        )
    elif shape_type == ShapeType.HEXAHEDRON:
        return ihexahedron.get_hexahedron_quadrature_weights(
            vertices, integration_order, quadrature_type=quadrature_type
        )
    elif shape_type == ShapeType.POLYHEDRON:
        return ipolyhedron.get_polyhedron_quadrature_weights(
            vertices, connectivity, integration_order, quadrature_type=quadrature_type
        )
    else:
        raise GeometryError("no such shape as {}".format(shape_type.value))


class Shape:
    type: ShapeType
    vertices: ndarray
    connectivity: Union[List[List[int]], None]

    def __init__(self, shape_type: ShapeType, shape_vertices: ndarray, connectivity: List[List[int]] = None):
        """

        Args:
            shape_type:
            shape_vertices:
            connectivity:
        """
        _check_shape(shape_type, shape_vertices)
        self.type = shape_type
        self.vertices = shape_vertices
        self.connectivity = connectivity

    def get_bounding_box(self) -> ndarray:
        return get_shape_bounding_box(self.vertices)


    def get_face_bounding_box(self) -> ndarray:
        rot = self.get_rotation_matrix()
        proj_v = (rot @ self.vertices)[:-1,:]
        return get_shape_bounding_box(proj_v)

    def get_centroid(self) -> ndarray:
        """

        Returns:

        """
        return get_centroid(self.type, self.vertices)

    def get_volume(self) -> float:
        """

        Returns:

        """
        return get_volume(self.type, self.vertices)

    def get_diameter(self) -> float:
        """

        Returns:

        """
        return get_diameter(self.type, self.vertices)

    def get_rotation_matrix(self) -> ndarray:
        """

        Returns:

        """
        return get_rotation_matrix(self.type, self.vertices)
        # return -get_rotation_matrix(self.type, self.vertices)

    def get_quadrature_size(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> int:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        return get_quadrature_size(
            self.type, self.vertices, integration_order, connectivity=self.connectivity, quadrature_type=quadrature_type
        )

    def get_quadrature_points(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> ndarray:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        return get_quadrature_points(
            self.type, self.vertices, integration_order, connectivity=self.connectivity, quadrature_type=quadrature_type
        )

    def get_quadrature_weights(
        self, integration_order: int, quadrature_type: QuadratureType = QuadratureType.GAUSS
    ) -> ndarray:
        """

        Args:
            integration_order:
            quadrature_type:

        Returns:

        """
        return get_quadrature_weights(
            self.type, self.vertices, integration_order, connectivity=self.connectivity, quadrature_type=quadrature_type
        )
