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

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation
from h2o.problem.resolution.exact import solve_newton_exact


class TestMecha(TestCase):
    def test_signorini(self):
        # --- VALUES
        u_min = 0.0
        u_max = 50.0 + 8.0 * 50.0
        ts = np.linspace(u_min, u_max, 100)
        time_steps = list(ts)
        iterations = 10

        # --- LOAD
        def volumetric_load(time: float, position: ndarray):
            return 0.0

        loads = [Load(volumetric_load, 0), Load(volumetric_load, 1), Load(volumetric_load, 2)]

        # --- BC
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed(time: float, position: ndarray) -> float:
            return 0.0

        boundary_conditions = [
            BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("TOP", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("TOP", fixed, BoundaryType.DISPLACEMENT, 2),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 2),
        ]

        # --- MESH
        # mesh_file_path = "meshes/signorini_coarse.msh"
        mesh_file_path = "meshes/signorini_fine.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN)

        # --- FINITE ELEMENT
        finite_element = FiniteElement(
            element_type=ElementType.HHO_EQUAL,
            polynomial_order=3,
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
            res_folder_path=get_current_res_folder_path() + "_80_PERCENT_DEPL"
        )

        # --- MATERIAL

        C10 = 2.668
        C01 = 0.271
        C20 = 0.466
        # K = 2939
        xnu = 0.499
        K = 4 * (1 + xnu) * (C10 + C01) / 3 / (1 - (2 * xnu))
        xyoun = 4 * (1 + xnu) * (C10 + C01);
        # parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
        parameters = {"YoungModulus": xyoun, "PoissonRatio": xnu}
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        print("STAB : {}".format(stabilization_parameter))
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="Signorini",
            hypothesis=mgis_bv.Hypothesis.TRIDIMENSIONAL,
            stabilization_parameter=stabilization_parameter,
            # lagrange_parameter=parameters["YoungModulus"],
            lagrange_parameter=stabilization_parameter,
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
