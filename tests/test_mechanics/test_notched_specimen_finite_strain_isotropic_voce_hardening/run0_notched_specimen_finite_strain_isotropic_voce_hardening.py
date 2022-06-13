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
from h2o.problem.resolution.local_equilibrium import solve_newton_local_equilibrium


class TestMecha(TestCase):
    def test_notched_specimen_finite_strain_linear_voce_hardening2(self):
        # --- VALUES
        u_min = 0.0
        u_mid = 0.00046
        u_mid2 = 0.0006
        u_max = 0.0008
        # time_steps = np.linspace(u_min, u_mid, 30)
        time_steps = np.concatenate(
            (np.linspace(u_min, u_mid, 30, endpoint=False), np.linspace(u_mid, u_max, 50, endpoint=True)))
        # time_steps = np.linspace(u_min, 2.3901563912771447E+09/0.0054, steps)
        ts0: list = list(np.linspace(u_min, u_mid, 30, endpoint=False))
        ts1: list = list(np.linspace(u_mid, u_mid2, 30, endpoint=False))
        ts2: list = list(np.linspace(u_mid2, u_max, 30, endpoint=True))
        time_steps = ts0 + ts1 + ts2
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
            # BoundaryCondition("TOP", pull, BoundaryType.PRESSURE, 1),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        mesh_file_path = "meshes/ssna.geof"
        # mesh_file_path = "meshes/ssna303_strcut_qua_2.msh"
        # mesh_file_path = "meshes/ssna303_strcut_tri_0.msh"
        # mesh_file_path = "meshes/ssna_quad_light.msh"
        # mesh_file_path = "meshes/ssna_tri_light.msh"
        mesh_file_path = "meshes/ssna_quad_hgh.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN)

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
            res_folder_path=get_current_res_folder_path() + "_STATIC_FINEMESH_HHO11"
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 206.9e9, "PoissonRatio": 0.29}
        coef = 1.0
        stabilization_parameter = coef * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            library_name="Voce",
            # library_name="FiniteStrainIsotropicLinearHardeningPlasticity",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=1.0 * parameters["YoungModulus"],
            # lagrange_parameter=1.0,
            field=displacement,
            parameters=parameters,
        )

        # --- SOLVE
        solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        # solve_newton_local_equilibrium(p, mat, verbose=False, debug_mode=DebugMode.NONE)

        from os import walk, path
        import matplotlib.pyplot as plt

        plt.plot(boundary_conditions[1].time_values, boundary_conditions[1].force_values)
        plt.grid()
        plt.show()
