import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from h2o.problem.solve.solve_condensation_axi import solve_condensation
from h2o.problem.solve.solve_implicit_axi import solve_implicit

from mgis import behaviour as mgis_bv

computation_parameters = [
    # --- ORDER 1
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 1, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    {"mesh" : "coarse", "order": 1, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 2},
    {"mesh" : "coarse", "order": 1, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 1, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    # --- ORDER 2
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 2, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    {"mesh" : "coarse", "order": 2, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 2},
    {"mesh" : "coarse", "order": 2, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 2, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    # --- ORDER 3
    # CONSISTENT JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 3, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator, "acceleration": 0},
    # ELASTIC JACOBIAN
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
    {"mesh" : "coarse", "order": 3, "algorithm" : "explicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 2},
    {"mesh" : "coarse", "order": 3, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 0},
    {"mesh" : "coarse", "order": 3, "algorithm" : "implicit", "integration" : mgis_bv.IntegrationType.IntegrationWithElasticOperator, "acceleration": 1},
]

for computation in computation_parameters:

    # --- VALUES
    time_steps = np.linspace(0.0, 0.0008, 100)
    iterations = 200

    # --- LOAD
    def volumetric_load(time: float, position: ndarray):
        return 0

    loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

    # --- BC
    def pull(time: float, position: ndarray) -> float:
        return time

    def clamped(time: float, position: ndarray) -> float:
        return 0.0

    boundary_conditions = [
        BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
        BoundaryCondition("BOTTOM", clamped, BoundaryType.DISPLACEMENT, 1),
        BoundaryCondition("LEFT", clamped, BoundaryType.DISPLACEMENT, 0),
    ]

    # --- MESH
    mesh_file_path = "meshes/notched_rod/notched_rod_{}.msh".format(computation["mesh"])

    # --- FIELD
    displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC)

    # --- FINITE ELEMENT
    finite_element = FiniteElement(
        element_type=ElementType.HHO_HIGH,
        polynomial_order=computation["order"],
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
        res_folder_path=get_current_res_folder_path() + "_notched_rod__{}__{}__acceleration_{}".format(computation["algorithm"], computation["integration"], computation["acceleration"])
    )

    # --- MATERIAL
    parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
    stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
    mat = Material(
        nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
        library_path="behaviours/bhv_large_strain_voce_plasticity/src/libBehaviour.so",
        library_name="LargeStrainVocePlasticity",
        hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
        stabilization_parameter=stabilization_parameter,
        lagrange_parameter=parameters["YoungModulus"],
        field=displacement,
        integration_type=computation["integration"],
        parameters=None,
    )

    # --- SOLVE
    if computation["algorithm"] == "explicit":
        solve_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE, accelerate=computation["acceleration"]) #0 -> no acceleration, 1 -> global acceleration, 2 -> local acceleration
    elif computation["algorithm"] == "implicit":
        solve_implicit(p, mat, verbose=False, debug_mode=DebugMode.NONE, accelerate=computation["acceleration"])
    else:
        raise SyntaxError("algorithm not known")
    