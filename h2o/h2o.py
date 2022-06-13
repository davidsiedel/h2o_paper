from enum import Enum, auto
from typing import List, Dict, Tuple, Callable, Union
import numpy as np
from numpy import ndarray
import pathlib
import shutil
import time
import os


class DebugMode(Enum):
    NONE = auto()
    LIGHT = auto()
    TWO = auto()


class BoundaryType(Enum):
    DISPLACEMENT = auto()
    PRESSURE = auto()
    SLIDE = auto()

class DomainType(Enum):
    POINT = auto()
    CURVE = auto()
    SURFACE = auto()
    VOLUME = auto()

class ShapeType(Enum):
    # POINT = auto()
    # SEGMENT = auto()
    # TRIANGLE = auto()
    # QUADRANGLE = auto()
    # POLYGON = auto()
    # TETRAHEDRON = auto()
    # HEXAHEDRON = auto()
    # POLYHEDRON = auto()
    SEGMENT = auto()
    TRIANGLE = auto()
    QUADRANGLE = auto()
    POLYGON = auto()
    TETRAHEDRON = auto()
    HEXAHEDRON = auto()
    POLYHEDRON = auto()
    PRISM = auto()
    PYRAMID = auto()
    POINT = auto()

class IterationOutput(Enum):
    CONVERGENCE = auto()
    INTEGRATION_FAILURE = auto()
    SYSTEM_SOLVED = auto()
    RESIDUAL_EVALUATED = auto()

class ExternalVariable(Enum):
    STATE_VARIABLE = auto()
    FIELD_VARIABLE = auto()

class QuadratureType(Enum):
    GAUSS = auto()


class BasisType(Enum):
    MONOMIAL = auto()


class ElementType(Enum):
    HDG_LOW = auto()
    HDG_EQUAL = auto()
    HDG_HIGH = auto()
    HHO_LOW = auto()
    HHO_EQUAL = auto()
    HHO_HIGH = auto()


class FieldType(Enum):
    SCALAR_PLANE = auto()
    DISPLACEMENT_LARGE_STRAIN = auto()
    DISPLACEMENT_SMALL_STRAIN = auto()
    DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN = auto()
    DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN = auto()
    DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS = auto()
    DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS = auto()
    DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC = auto()
    DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC = auto()


class FluxType(Enum):
    STRESS_PK1 = auto()
    STRESS_CAUCHY = auto()


class GradType(Enum):
    DISPLACEMENT_TRANSFORMATION_GRADIENT = auto()
    DISPLACEMENT_SMALL_STRAIN = auto()


class DerivationType(Enum):
    SYMMETRIC = auto()
    REGULAR = auto()
    SMALL_STRAIN_AXISYMMETRIC = auto()
    LARGE_STRAIN_AXISYMMETRIC = auto()

class SolverType(Enum):
    STATIC_CONDENSATION = auto()
    CELL_EQUILIBRIUM = auto()


class GeometryError(Exception):
    pass


class QuadratureError(Exception):
    pass


class ElementError(Exception):
    pass


def get_project_path():
    return pathlib.Path(__file__).parent.parent.absolute()


def get_res_file_path(res_file_name: str, suffix: str):
    project_path = get_project_path()
    return os.path.join(project_path, "res/{}_{}.txt".format(res_file_name, suffix))


def get_current_res_folder_path() -> str:
    res_path = os.path.join(os.getcwd(), "res")
    return res_path


# real = np.float64
real = float
# real = np.float32
# intg = np.uint8
intg = int
size_type = np.uint8
debug_mode = DebugMode.NONE