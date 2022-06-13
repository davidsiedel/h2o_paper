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

# from h2o.problem.resolution.solve_linear_system import solve_newton_static_condensation
from h2o.problem.resolution.static_condensation_axisymmetric import solve_newton_static_condensation
from h2o.problem.resolution.exact import solve_newton_exact


class TestMecha(TestCase):
    def test_axisymmetric(self):
        # --- VALUES
        ts = np.linspace(0.0, 0.1/2.0, 4)
        ts = np.linspace(0.0, 1.0e-3, 20)
        time_steps = list(ts)
        iterations = 10

        # --- LOAD
        def volumetric_load(time: float, position: ndarray):
            return 0.0

        def fr(time: float, position: ndarray):
            part0 = -(2.0 * (np.pi) ** 2 + 1.0 / (position[0] ** 2)) * np.sin(np.pi * position[0]) * np.sin(np.pi * position[1])
            part1 = (np.pi / position[0]) * np.cos(np.pi * position[0]) * np.cos(np.pi * position[1])
            # return (part0 + part1) * time
            return 0.0

        def fz(time: float, position: ndarray):
            part0 = -(2.0 * (np.pi) ** 2) * np.cos(np.pi * position[0]) * np.cos(np.pi * position[1])
            part1 = (np.pi / position[0]) * np.sin(np.pi * position[0]) * np.sin(np.pi * position[1])
            # return (part0 - part1) * time
            return 0.0

        loads = [Load(fr, 0), Load(fz, 1)]

        # --- BC
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed_sin(time: float, position: ndarray) -> float:
            return 0.0

        def pressure_in(time: float, position: ndarray) -> float:
            return 300.e6 * time

        def pressure_out(time: float, position: ndarray) -> float:
            return - 100.e6 * time

        # boundary_conditions
        #     BoundaryCondition("TOP", fixed_sin, BoundaryType.DISPLACEMENT, 0),
        #     BoundaryCondition("BOTTOM", fixed_sin, BoundaryType.DISPLACEMENT, 0),
        #     BoundaryCondition("LEFT", fixed_sin, BoundaryType.DISPLACEMENT, 0),
        #     BoundaryCondition("RIGHT", fixed_sin, BoundaryType.DISPLACEMENT, 0),
        #     BoundaryCondition("TOP", fixed_cos, BoundaryType.DISPLACEMENT, 1),
        #     BoundaryCondition("BOTTOM", fixed_cos, BoundaryType.DISPLACEMENT, 1),
        #     BoundaryCondition("LEFT", fixed_cos, BoundaryType.DISPLACEMENT, 1),
        #     BoundaryCondition("RIGHT", fixed_cos, BoundaryType.DISPLACEMENT, 1),
        # ]

        boundary_conditions = [
            BoundaryCondition("TOP", fixed_sin, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("BOTTOM", fixed_sin, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", pressure_in, BoundaryType.PRESSURE, 0),
            BoundaryCondition("RIGHT", pressure_out, BoundaryType.PRESSURE, 0),
        ]

        # boundary_conditions = [
        #     BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
        #     BoundaryCondition("BOTTOM", fixed_sin, BoundaryType.DISPLACEMENT, 1),
        #     BoundaryCondition("LEFT", fixed_sin, BoundaryType.PRESSURE, 0),
        # ]

        # --- MESH
        mesh_file_path = "meshes/cylinder.msh"
        # mesh_file_path = "meshes/unit_square.msh"
        # mesh_file_path = "meshes/unit_square_10.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC)

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
            tolerance=1.0e-8,
            res_folder_path=get_current_res_folder_path() + "_voce_large_strain"
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 206.9e6, "PoissonRatio": 0.29}
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        print("STAB : {}".format(stabilization_parameter))
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour_voce_large_strain_mpa/src/libBehaviour.so",
            library_name="Voce",
            hypothesis=mgis_bv.Hypothesis.AXISYMMETRICAL,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=stabilization_parameter,
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
