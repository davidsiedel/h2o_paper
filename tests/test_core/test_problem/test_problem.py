from unittest import TestCase

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation


class TestProblem(TestCase):
    def test_problem_build(self, verbose=True):
        # --- VALUES
        p_min = 0.0
        p_max = 1. / 16.
        time_steps = np.linspace(p_min, p_max, 10)
        iterations = 100

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
            BoundaryCondition("RIGHT", pull, BoundaryType.PRESSURE, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
        ]

        # --- MESH
        mesh_file_path = (
            "meshes/triang_r.geof"
        )

        # --- FIELD
        displacement = Field(
            label="U",
            field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
        )

        # --- FINITE ELEMENT
        finite_element = FiniteElement(
            element_type=ElementType.HDG_EQUAL, polynomial_order=1, euclidean_dimension=2, basis_type=BasisType.MONOMIAL
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
            tolerance=1.0e-4,
            res_folder_path=get_current_res_folder_path()
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 70.0e9, "PoissonRatio": 0.34}
        stabilization_parameter = parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="../../test_mechanics/test_element/2D/test_2D_small_strain_linear_elasticity/behaviour_elasticity_small_strain/src/libBehaviour.so",
            library_name="Elasticity",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
            # finite_strains=False
        )

        # --- SOLVE
        solve_newton_2(p, mat)
