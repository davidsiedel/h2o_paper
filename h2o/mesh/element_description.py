from typing import List
from h2o.h2o import *


class CellDescription:
    shape_type: ShapeType
    face_vertices_ordering: List[List[int]]
    number_of_vertices: int
    number_of_faces: int
    euclidean_dimension: int
    cell_hash: str

    def __init__(
        self,
        shape_type: ShapeType,
        face_vertices_ordering: List[List[int]],
        number_of_vertices: int,
        number_of_faces: int,
        euclidean_dimension: int,
        label: str = None,
    ):
        """

        Args:
            shape_type:
            face_vertices_ordering:
            number_of_vertices:
            number_of_faces:
            euclidean_dimension:
            label:
        """
        self.shape_type = shape_type
        self.face_vertices_ordering = face_vertices_ordering
        self.number_of_vertices = number_of_vertices
        self.number_of_faces = number_of_faces
        self.euclidean_dimension = euclidean_dimension
        if label is None:
            label = ""
            for row in range(len(face_vertices_ordering)):
                for col in range(len(face_vertices_ordering)):
                    label.join("r{}c{}i{}".format(row, col, face_vertices_ordering[row][col]))
        self.cell_hash = label

    def __hash__(self):
        """

        Returns:

        """
        return hash(self.cell_hash)

    def __eq__(self, other):
        """

        Args:
            other:

        Returns:

        """
        return self.face_vertices_ordering == other.face_vertices_ordering

    def __ne__(self, other):
        """

        Args:
            other:

        Returns:

        """
        return not (self.face_vertices_ordering == other.face_vertices_ordering)

    def __repr__(self):
        """

        Returns:

        """
        repr = "{} : {}\n{} : {}\n{} : {}".format(
            ShapeType.__name__,
            self.shape_type.value,
            self.__dir__()[2],
            self.number_of_vertices,
            self.__dir__()[3],
            self.number_of_faces,
        )
        return "%s(\n%s\n)" % (type(self).__name__, repr)


class FaceDescription:
    shape_type: ShapeType
    number_of_vertices: int
    manifold_dimension: int
    face_hash: str

    def __init__(
        self, shape_type: ShapeType, number_of_vertices: int, euclidean_dimension: int, label: str = None,
    ):
        """

        Args:
            shape_type:
            number_of_vertices:
            euclidean_dimension:
            label:
        """
        self.shape_type = shape_type
        self.number_of_vertices = number_of_vertices
        self.manifold_dimension = euclidean_dimension - 1
        if label is None:
            self.face_hash = "{}".format(number_of_vertices)
        else:
            self.face_hash = label

    def __hash__(self):
        """

        Returns:

        """
        return hash(self.face_hash)

    def __eq__(self, other):
        """

        Args:
            other:

        Returns:

        """
        return self.face_hash == other.face_hash

    def __ne__(self, other):
        """

        Args:
            other:

        Returns:

        """
        return not (self.face_hash == other.face_hash)

    def __repr__(self):
        """

        Returns:

        """
        repr = "{} : {}\n{} : {}".format(
            ShapeType.__name__, self.shape_type.value, self.__dir__()[1], self.number_of_vertices,
        )
        return "%s(\n%s\n)" % (type(self).__name__, repr)
