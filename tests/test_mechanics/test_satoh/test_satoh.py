from unittest import TestCase

import numpy as np

from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.solve_static_condensation_thermo import solve_newton_static_condensation
from h2o.problem.resolution.exact import solve_newton_exact


class TestMecha(TestCase):
    def test_signorini(self):
        # --- VALUES
        t_min = 293.15
        t_max = 293.15 + 2000.0
        ts = np.linspace(t_min, t_max, 10)
        time_steps = list(ts)
        iterations = 10

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
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("RIGHT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("RIGHT", fixed, BoundaryType.DISPLACEMENT, 1),
        ]

        # --- MESH
        # mesh_file_path = "meshes/signorini_coarse.msh"
        mesh_file_path = "meshes/satoh2.msh"

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
            res_folder_path=get_current_res_folder_path() + "_2000DEG_NON_HOMOGENEOUS_TEMP_PARABOLIC"
        )

        # --- MATERIAL

        parameters = {"YoungModulus": 200.e9, "PoissonRatio": 0.4999}
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        print("STAB : {}".format(stabilization_parameter))
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="Elasticity",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            # lagrange_parameter=parameters["YoungModulus"],
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
