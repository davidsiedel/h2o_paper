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

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation as solve_cartesian
from h2o.problem.resolution.static_condensation_axisymmetric import solve_newton_static_condensation as solve_axi
# from h2o.problem.resolution.exact import solve_newton_exact


class TestMecha(TestCase):
    def test_sphere_3D(self):
        # --- VALUES
        ts = np.linspace(0.0, 0.1/2.0, 4)
        ts = np.linspace(0.0, 8.e6, 30)
        ts = np.linspace(0.0, 2.540262099505798e6, 20)
        ts = np.linspace(0.0, 3.1753276243822475e6, 20)
        time_steps = list(ts)
        iterations = 10

        # --- LOAD
        def volumetric_load(time: float, position: ndarray):
            return 0.0

        def fr(time: float, position: ndarray):
            return 0.0

        def fz(time: float, position: ndarray):
            return 0.0

        loads = [Load(fr, 0), Load(fz, 1), Load(fz, 2)]
        radius = 0.8

        # --- BC
        def pull(time: float, position: ndarray) -> float:
            return time
        def pull_x(time: float, position: ndarray) -> float:
            r = np.sqrt(position[0] ** 2 + position[1] ** 2)
            cos_theta = position[0] / r
            sin_theta = position[1] / r
            cos_phi = position[2] / np.linalg.norm(position)
            sin_phi = r / np.linalg.norm(position)
            x = radius * sin_phi * cos_theta
            return time * x
        def pull_y(time: float, position: ndarray) -> float:
            r = np.sqrt(position[0] ** 2 + position[1] ** 2)
            cos_theta = position[0] / r
            sin_theta = position[1] / r
            cos_phi = position[2] / np.linalg.norm(position)
            sin_phi = r / np.linalg.norm(position)
            y = radius * sin_phi * sin_theta
            return time * y
        def pull_z(time: float, position: ndarray) -> float:
            r = np.sqrt(position[0] ** 2 + position[1] ** 2)
            cos_theta = position[0] / r
            sin_theta = position[1] / r
            cos_phi = position[2] / np.linalg.norm(position)
            sin_phi = r / np.linalg.norm(position)
            z = radius * cos_phi
            return time * z

        def fixed_sin(time: float, position: ndarray) -> float:
            return 0.0

        boundary_conditions = [
            BoundaryCondition("YZ", fixed_sin, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("XZ", fixed_sin, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("XY", fixed_sin, BoundaryType.DISPLACEMENT, 2),
            BoundaryCondition("INTERIOR", pull_x, BoundaryType.PRESSURE, 0),
            BoundaryCondition("INTERIOR", pull_y, BoundaryType.PRESSURE, 1),
            BoundaryCondition("INTERIOR", pull_z, BoundaryType.PRESSURE, 2),
        ]

        # --- MESH
        mesh_file_path = "meshes/sphere_3D_m.msh"
        mesh_file_path = "meshes/sphere_3D.msh"
        # mesh_file_path = "meshes/unit_square.msh"
        # mesh_file_path = "meshes/unit_square_10.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN)

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
            res_folder_path=get_current_res_folder_path() + "_3D_SMALL_VOCE"
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
        parameters = {"YoungModulus": 28.85e6, "PoissonRatio": 0.499}
        parameters = {"YoungModulus": 28.85e6, "PoissonRatio": 0.499}

        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        print("STAB : {}".format(stabilization_parameter))
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour_perfect_plasticity_large_strain/src/libBehaviour.so",
            library_name="Voce",
            hypothesis=mgis_bv.Hypothesis.TRIDIMENSIONAL,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=stabilization_parameter,
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        # solve_axi(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        solve_cartesian(p, mat, verbose=False, debug_mode=DebugMode.NONE)
