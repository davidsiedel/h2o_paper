import time

import numpy as np

from h2o.problem.problem import Problem, clean_res_dir
from h2o.problem.material import Material
from h2o.fem.element.element import Element
from h2o.h2o import *

from mgis import behaviour as mgis_bv
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sys
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=1000, nanstr='nan', precision=2,suppress=True, threshold=sys.maxsize, formatter=None)

from h2o.problem.output import create_output_txt


def get_traction_force(problem: Problem, material: Material, verbose: bool = False, debug_mode: DebugMode = DebugMode.NONE):
    clean_res_dir(problem.res_folder_path)

def get_traction_force2(element: Element, material: Material, element_index: int):
    return
