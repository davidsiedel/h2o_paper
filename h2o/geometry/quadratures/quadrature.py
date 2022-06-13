from h2o.h2o import *


class QuadratureItem(Enum):
    POINTS = auto()
    WEIGHTS = auto()
    JACOBIAN = auto()
    SIZE = auto()
