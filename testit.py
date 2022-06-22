import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation as solve_cartesian
from h2o.problem.resolution.solve_static_condensation_thermo import solve_newton_static_condensation as solve_axi
from h2o.problem.resolution.solve_static_condensation_thermo_axi import solve_newton_static_condensation as solve_axi
# from h2o.problem.resolution.exact import solve_newton_exact

ts = np.linspace(0.0, 0.1/2.0, 4)
ts = np.linspace(0.0, 1.0, 2)
time_steps = list(ts)
iterations = 10

# --- LOAD
def volumetric_load(time: float, position: ndarray):
    return 0.0

def fr(time: float, position: ndarray):
    return 0.0

def fz(time: float, position: ndarray):
    return 0.0

loads = [Load(fr, 0), Load(fz, 1)]

# --- BC
def pull(time: float, position: ndarray) -> float:
    return time

def fixed_sin(time: float, position: ndarray) -> float:
    return 0.0

boundary_conditions = [
    BoundaryCondition("LEFT", fixed_sin, BoundaryType.DISPLACEMENT, 0),
    # BoundaryCondition("LEFT", fixed_sin, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("RIGHT", fixed_sin, BoundaryType.DISPLACEMENT, 0),
    BoundaryCondition("BOTTOM", fixed_sin, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("RIGHT", fixed_sin, BoundaryType.DISPLACEMENT, 1),
]

# --- MESH
mesh_file_path = "meshes/rod_half.msh"
mesh_file_path = "meshes/satoh_20.msh"
# mesh_file_path = "meshes/rod_half_20.msh"
mesh_file_path = "meshes/satoh_10.msh"
# mesh_file_path = "meshes/unit_square.msh"
# mesh_file_path = "meshes/unit_square_10.msh"

# --- FIELD
displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC)

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
    res_folder_path=get_current_res_folder_path() + "_AXI_SATOH_INCOMP_1_AXI"
)

# --- MATERIAL
# parameters = {"YoungModulus": 200.0e6, "PoissonRatio": 0.499}
parameters = {"YoungModulus": 150.0e9, "PoissonRatio": 0.4999}
stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
print("STAB : {}".format(stabilization_parameter))
mat = Material(
    nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
    library_path="behaviours/bhv_linear_thermo_elasticity/src/libBehaviour.so",
    library_name="LinearThermoElasticity",
    hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
    stabilization_parameter=stabilization_parameter,
    lagrange_parameter=stabilization_parameter,
    field=displacement,
    parameters=None,
)

# --- SOLVE
solve_axi(p, mat, verbose=False, debug_mode=DebugMode.NONE)
# solve_cartesian(p, mat, verbose=False, debug_mode=DebugMode.NONE)
