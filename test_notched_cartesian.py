# from unittest import TestCase

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.solve.solve_implicit_new import solve_implicit
from h2o.problem.solve.solve_condensation import solve_condensation

# --- VALUES
u_min = 0.0
u_max = 0.0008
steps = 50
time_steps = np.linspace(u_min, u_max, steps)
iterations = 10
print(list(time_steps))

# --- LOAD
def volumetric_load(time: float, position: ndarray):
    return 0

loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

# --- BC
def pull(time: float, position: ndarray) -> float:
    return time

def fixed(time: float, position: ndarray) -> float:
    return 0.0

boundary_conditions = [
    # BoundaryCondition("LRU", pull, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("LRD", fixed, BoundaryType.DISPLACEMENT, 1),
    # BoundaryCondition("AXESYM", fixed, BoundaryType.DISPLACEMENT, 0),
    BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
    BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
    BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
]

# --- MESH
# mesh_file_path = "meshes/ssna.geof"
mesh_file_path = "meshes/ssna_quad_light.msh"
# mesh_file_path = "meshes/ssna_quad_mid.msh"
# mesh_file_path = "meshes/ssna303_COMP_AXI_on_axis.msh"

# --- FIELD
displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC)

# --- FINITE ELEMENT
finite_element = FiniteElement(
    element_type=ElementType.HHO_HIGH,
    polynomial_order=3,
    euclidean_dimension=displacement.euclidean_dimension,
    basis_type=BasisType.MONOMIAL,
)

algorithm_type = "STATIC"
algorithm_type = "IMPLICIT"

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
    tolerance=1.0e-12,
    res_folder_path=get_current_res_folder_path() + "_hho_high_3"
)

# --- MATERIAL
parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
mat = Material(
    nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
    library_path="behaviour/src/libBehaviour.so",
    library_name="Voce",
    # library_name="FiniteStrainIsotropicLinearHardeningPlasticity",
    hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
    stabilization_parameter=stabilization_parameter,
    lagrange_parameter=parameters["YoungModulus"],
    field=displacement,
    parameters=None,
)

# --- SOLVE
if algorithm_type == "STATIC":
    solve_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
elif algorithm_type == "IMPLICIT":
    solve_implicit(p, mat, verbose=False, debug_mode=DebugMode.NONE)
