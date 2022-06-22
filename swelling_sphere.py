import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from h2o.problem.solve.solve_implicit_axi import solve_implicit
from h2o.problem.solve.solve_condensation_axi import solve_condensation

from mgis import behaviour as mgis_bv

# --- VALUES
time_steps = np.linspace(0.0, 0.20238839836127986, 40, endpoint=True)
iterations = 100

# --- LOAD
def fr(time: float, position: ndarray):
    return 0.0

def fz(time: float, position: ndarray):
    return 0.0

loads = [Load(fr, 0), Load(fz, 1)]

# --- BC
def swell_x(time: float, position: ndarray) -> float:
    return time * position[0]
def swell_y(time: float, position: ndarray) -> float:
    return time * position[1]

def clamped(time: float, position: ndarray) -> float:
    return 0.0

boundary_conditions = [
    BoundaryCondition("TOP", clamped, BoundaryType.DISPLACEMENT, 0),
    BoundaryCondition("BOTTOM", clamped, BoundaryType.DISPLACEMENT, 1),
    BoundaryCondition("INTERIOR", swell_x, BoundaryType.DISPLACEMENT, 0),
    BoundaryCondition("INTERIOR", swell_y, BoundaryType.DISPLACEMENT, 1),
]

# --- MESH
mesh_file_path = "meshes/sphere/sphere_coarse.msh"
# mesh_file_path = "meshes/sphere_fine.msh"
# mesh_file_path = "meshes/unit_square.msh"
# mesh_file_path = "meshes/unit_square_10.msh"

# --- FIELD
# displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC)
displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC)

# --- FINITE ELEMENT
finite_element = FiniteElement(
    element_type=ElementType.HHO_HIGH,
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
    res_folder_path=get_current_res_folder_path() + "_swelling_sphere"
)

# --- MATERIAL
parameters = {"YoungModulus": 28.85e6, "PoissonRatio": 0.499}
stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
print("STAB : {}".format(stabilization_parameter))
mat = Material(
    nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
    library_path="behaviours/bhv_large_strain_perfect_plasticity/src/libBehaviour.so",
    library_name="LargeStrainPerfectPlasticity",
    hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
    stabilization_parameter=stabilization_parameter,
    lagrange_parameter=stabilization_parameter,
    field=displacement,
    integration_type=mgis_bv.IntegrationType.IntegrationWithElasticOperator,
    # integration_type=mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator,
    parameters=None,
)

# --- SOLVE
# solve_implicit(p, mat, verbose=False, debug_mode=DebugMode.NONE, accelerate=True)
solve_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
