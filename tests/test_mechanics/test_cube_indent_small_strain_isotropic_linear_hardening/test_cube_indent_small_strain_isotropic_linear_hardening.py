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
    def test_cube_indent_small_strain_isotropic_linear_hardening(self):
        # --- MECHANICAL TEST
        # --- VALUES
        # On détermine ici le tableau des pas de temps. Dans le contexte de la mécanique quasi-statique, on néglige les
        # effets d'accélération, et on peut se peremttre de confondre le temps avec la variable de chargement (en
        # déplacement ou en pression). Le chargement de ce cas test est une pression imposée sur une partie de la face
        # supérieure du cube, on définit donc la pression du chargement final P_max, et la pression appliqué en début
        # de chargement P_min, afin de définir N pas de temps uniformément répartis entre P_min et P_max, avec N le
        # nombre de pas de temps.
        P_min = 0.
        P_ela = -180.e6
        P_god = -330.e6
        P_god = -310.e6
        P_max = -350.e6
        ts0 = np.linspace(P_min, P_ela, 5)
        ts1 = np.linspace(P_ela, P_god, 15)
        # ts2 = np.linspace(P_god, P_max, 200)
        # time_steps = list(ts0) + list(ts1) + list(ts2)
        time_steps = list(ts0) + list(ts1)
        print(time_steps)
        iterations = 10

        # --- LOAD
        # On définit les conditions de chargement volumiques; il s'agit d'une fonction de l'espace et du temps, que l'on
        # passe en argument au modèle. Dans le cas de la plaque entaillée, on néglige l'influence du poids propre de la
        # structure devant celle des conditions aux limites; on donne donc une charge volumique nulle, pour tout instant
        # et tout point de la strcuture.
        def volumetric_load(time: float, position: ndarray):
            return 0

        loads = [Load(volumetric_load, 0), Load(volumetric_load, 1)]

        # --- BC
        # On définit les conditions aux limites en
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed(time: float, position: ndarray) -> float:
            return 0.0

        boundary_conditions = [
            BoundaryCondition("INDENT", pull, BoundaryType.PRESSURE, 2),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 2),
            # BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            # BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 2),
            BoundaryCondition("FRONT_Y", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("FRONT_X", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        # mesh_file_path = "meshes/cube_indent.msh"
        # mesh_file_path = "meshes/maillage_grossier.msh"
        mesh_file_path = "meshes/cube_test.msh"
        mesh_file_path = "meshes/indented_cube_hexa.msh"
        mesh_file_path = "meshes/indented_cube_hexa_fine.msh"
        mesh_file_path = "meshes/indented_cube_tri.msh"
        # mesh_file_path = "meshes/try_0.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN)

        # --- FINITE ELEMENT
        finite_element = FiniteElement(
            element_type=ElementType.HDG_EQUAL,
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
            res_folder_path=get_current_res_folder_path()
        )

        # --- MATERIAL
        parameters = {"YoungModulus": 200.e9, "PoissonRatio": 0.3}
        # stabilization_parameter = 1000. * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        # stabilization_parameter = 0.0001 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        stabilization_parameter = 1.0 * parameters["YoungModulus"] / (1.0 + parameters["PoissonRatio"])
        mat = Material(
            nq=p.mesh.number_of_cell_quadrature_points_in_mesh,
            library_path="behaviour/src/libBehaviour.so",
            # library_name="Voce",
            library_name="FiniteStrainIsotropicLinearHardeningPlasticity",
            hypothesis=mgis_bv.Hypothesis.TRIDIMENSIONAL,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=parameters["YoungModulus"],
            field=displacement,
            parameters=None,
        )

        # --- SOLVE
        # solve_newton_static_condensation(p, mat, verbose=False, debug_mode=DebugMode.NONE)
        solve_newton_local_equilibrium(p, mat, verbose=False, debug_mode=DebugMode.NONE)

        # from pp.plot_ssna import plot_det_f
        #
        # # plot_det_f(46, "res")
        #
        # res_folder = "res"
        # from os import walk, path
        # import matplotlib.pyplot as plt
        # from matplotlib.colors import LinearSegmentedColormap
        #
        # def __plot(column: int, time_step_index: int):
        #
        #     _, _, filenames = next(walk(res_folder))
        #     # for time_step_index in range(1, len(time_steps)):
        #     # for time_step_index in range(30, len(time_steps)):
        #     for filename in filenames:
        #         if "{}".format(time_step_index).zfill(6) in filename and "qdp" in filename:
        #             hho_file_path = path.join(res_folder, filename)
        #             with open(hho_file_path, "r") as hho_res_file:
        #                 fig, ax0d = plt.subplots(nrows=1, ncols=1)
        #                 c_hho = hho_res_file.readlines()
        #                 field_label = c_hho[0].split(",")[column]
        #                 number_of_points = len(c_hho) - 1
        #                 # for _iloc in range(len(c_hho)):
        #                 #     line = c_hho[_iloc]
        #                 #     x_coordinates = float(line.split(",")[0])
        #                 #     y_coordinates = float(line.split(",")[1])
        #                 #     if (x_coordinates - 0.0) ** 2 + (y_coordinates)
        #                 eucli_d = displacement.euclidean_dimension
        #                 points = np.zeros((eucli_d, number_of_points), dtype=real)
        #                 field_vals = np.zeros((number_of_points,), dtype=real)
        #                 for l_count, line in enumerate(c_hho[1:]):
        #                     x_coordinates = float(line.split(",")[0])
        #                     y_coordinates = float(line.split(",")[1])
        #                     field_value = float(line.split(",")[column])
        #                     points[0, l_count] += x_coordinates
        #                     points[1, l_count] += y_coordinates
        #                     field_vals[l_count] += field_value
        #                 x, y = points
        #                 colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        #                 perso = LinearSegmentedColormap.from_list("perso", colors, N=1000)
        #                 vmin = min(field_vals[:])
        #                 vmax = max(field_vals[:])
        #                 # vmin = -3.656e9
        #                 # vmax = 1.214e9
        #                 levels = np.linspace(vmin, vmax, 50, endpoint=True)
        #                 ticks = np.linspace(vmin, vmax, 10, endpoint=True)
        #                 datad = ax0d.tricontourf(x, y, field_vals[:], cmap=perso, levels=levels)
        #                 ax0d.get_xaxis().set_visible(False)
        #                 ax0d.get_yaxis().set_visible(False)
        #                 ax0d.set_xlabel("map of the domain $\Omega$")
        #                 cbar = fig.colorbar(datad, ax=ax0d, ticks=ticks)
        #                 cbar.set_label("{} : {}".format(field_label, time_step_index), rotation=270, labelpad=15.0)
        #                 # plt.savefig("/home/dsiedel/Projects/pythhon/plots/{}.png".format(time_step))
        #                 plt.show()
        #
        # for tsindex in [1, 50, 100, 192]:
        #     # __plot(15, tsindex)
        #     pass
        # # __plot(15, 19)
        # __plot(15, 22)
        # # __plot(3)
        #
        # # --- POST PROCESSING
        # # from pp.plot_data import plot_data
        # # mtest_file_path = "mtest/finite_strain_isotropic_linear_hardening.res"
        # # hho_res_dir_path = "../../../res"
        # # number_of_time_steps = len(time_steps)
        # # m_x_inedx = 1
        # # m_y_index = 6
        # # d_x_inedx = 4
        # # d_y_inedx = 9
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # # m_x_inedx = 1
        # # m_y_index = 7
        # # d_x_inedx = 4
        # # d_y_inedx = 10
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # # m_x_inedx = 1
        # # m_y_index = 8
        # # d_x_inedx = 4
        # # d_y_inedx = 11
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
        # # m_x_inedx = 1
        # # m_y_index = 9
        # # d_x_inedx = 4
        # # d_y_inedx = 12
        # # plot_data(mtest_file_path, hho_res_dir_path, number_of_time_steps, m_x_inedx, m_y_index, d_x_inedx, d_y_inedx)
