# from unittest import TestCase

# spack load tfel
# spack load mgis
# spack load py-numpy@1.21.3
# spack load py-scipy@1.8.0
# spack load py-matplotlib@3.4.3

import enum
import tfel.math
import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation
from h2o.problem.resolution.static_condensation import solve_newton_static_condensation
from h2o.problem.resolution.solve_generic import solve_newton
from h2o.problem.resolution.exact import solve_newton_exact
from h2o.problem.resolution.local_equilibrium import solve_newton_local_equilibrium
from h2o.problem.resolution.local_equilibrium2 import solve_newton_local_equilibrium2
from h2o.problem.resolution.local_equilibrium_3 import solve_newton_local_equilibrium3
from h2o.problem.resolution.local_equilibrium_4 import solve_newton_local_equilibrium4
from h2o.problem.solve.solve_implicit import solve_implicit
from h2o.problem.solve.solve_condensation import solve_condensation

# --- VALUES
P_min = 0.0
P_mid = 1.4e8
P_max = 5.e6 / (16.e-3)
time_steps = np.linspace(P_min, P_max, 10)
iterations = 500

# --- LOAD
def volumetric_load(time: float, position: ndarray):
    return 0.0

loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

# --- BC
def pull(time: float, position: ndarray) -> float:
    return time

def fixed(time: float, position: ndarray) -> float:
    return 0.0

boundary_conditions = [
    BoundaryCondition("RIGHT", pull, BoundaryType.PRESSURE, 1),
    BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
    BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
]

# --- MESH
# mesh_file_path = "meshes/cook_5.geof"
# mesh_file_path = "meshes/cook_30.geof"
mesh_file_path = "meshes/cook_quadrangles_0.msh"
mesh_file_path = "meshes/cook_10_quadrangles_structured.msh"
mesh_file_path = "meshes/cook_20_quadrangles_structured.msh"
# mesh_file_path = "meshes/cook_01_quadrangles_structured.msh"
# mesh_file_path = "meshes/cook_02_quadrangles_structured.msh"
# mesh_file_path = "meshes/cook_03_quadrangles_structured.msh"
# mesh_file_path = "meshes/cook_10_triangles_structured.msh"
# mesh_file_path = "meshes/cook_16_triangles_structured.msh"
# ----
# mesh_file_path = "meshes/cook_compare_m.geof"
mesh_file_path = "meshes/cook_32_triangles_structured.msh"
mesh_file_path = "meshes/cook_32_quadrangles_structured.msh"
mesh_file_path = "meshes/cook_20_quadrangles_structured.msh"
# mesh_file_path = "meshes2/cook_soda_45.msh"
# mesh_file_path = "meshes2/cook_45_quadrangles_structured2.msh"
mesh_file_path = "meshes/cook_10_quadrangles_structured.msh"

# --- FIELD
displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)

# --- FINITE ELEMENT
finite_element = FiniteElement(
    element_type=ElementType.HHO_EQUAL,
    polynomial_order=1,
    euclidean_dimension=displacement.euclidean_dimension,
    basis_type=BasisType.MONOMIAL,
)

# --- PROBLEM
p = Problem(
    mesh_file_path=mesh_file_path,
    field=displacement,
    finite_element=finite_element,
    time_steps=time_steps,
    iterations=iterations,
    boundary_conditions=boundary_conditions,
    loads=loads,
    quadrature_type=QuadratureType.GAUSS,
    tolerance=1.0e-6,
    res_folder_path=get_current_res_folder_path() + "_COOK ELASTICITY"
)

# --- MATERIAL
parameters = {"YoungModulus": 70.e9, "PoissonRatio": 0.3}
stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
mat = Material(
    nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
    library_path="behaviours/bhv_linear_elasticity/src/libBehaviour.so",
    library_name="Elasticity",
    hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
    stabilization_parameter=stabilization_parameter,
    lagrange_parameter=parameters["YoungModulus"],
    field=displacement,
    integration_type=mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator,
    parameters=None,
)

# --- SOLVE
solve_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
    