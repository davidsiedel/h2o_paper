from unittest import TestCase

import numpy as np

from h2o.problem.finite_element_field import FiniteElementField
from h2o.problem.displacement_finite_element_field import DisplacementFiniteElementField
from h2o.problem.damage_finite_element_field import DamageFiniteElementField
from h2o.problem.coupled_problem2 import CoupledProblem
from h2o.problem.problem import Problem
from h2o.problem.boundary_condition import BoundaryCondition
from h2o.problem.material import Material
from h2o.mesh.mesh import Mesh
from h2o.problem.load import Load
from h2o.field.field import Field
from h2o.fem.element.finite_element import FiniteElement
from h2o.fem.element.displacement_element import DisplacementElement
from h2o.h2o import *
from mgis import behaviour as mgis_bv

from h2o.problem.resolution.static_condensation import solve_newton_static_condensation
from h2o.problem.resolution.local_equilibrium import solve_newton_local_equilibrium


class TestMecha(TestCase):
    def test_heat_transfer(self):
        # --- VALUES
        d_min = 0.0
        d_max = 1.0
        time_steps: list = list(np.linspace(d_min, d_max, 2, endpoint=True))
        iterations = 10
        print(list(time_steps))

        # --- LOAD
        def volumetric_load(time: float, position: ndarray):
            return 0

        def meca_load(time: float, position: ndarray):
            # ret: float = ((4.0 * np.pi) ** 2) * np.sin(4.0 * np.pi * position[0])
            ret: float = 1.0 * position[0]
            return ret

        def sinus_load(time: float, position: ndarray):
            G_c: float = 1.0
            l_0: float = 0.1
            # ret: float = (G_c * (np.sin(4.0 * np.pi * position[0]) / 2.0) * (1.0 + l_0 ** 2)) / (2.0 * l_0 * (1.0 - (np.sin(4.0 * np.pi * position[0]) / 2.0)))
            ret: float = ((4.0 * np.pi) ** 2) * np.sin(4.0 * np.pi * position[0])
            ret: float = 1.0 / position[0]
            ret: float = 1.0
            return ret

        loads = [Load(meca_load, 0), Load(volumetric_load, 1)]

        # --- BC
        def pull(time: float, position: ndarray) -> float:
            return time

        def fixed(time: float, position: ndarray) -> float:
            return 0.0

        def damaged(time: float, position: ndarray) -> float:
            return 10. * time

        displacement_boundary_conditions = [
            BoundaryCondition("RIGHT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        damage_boundary_conditions = [
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("RIGHT", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("TOP", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- FIELD
        displacement_field = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)
        damage_field = Field(label="D", field_type=FieldType.SCALAR_PLANE)

        # --- FINITE ELEMENT
        displacement_finite_element = FiniteElement(
            element_type=ElementType.HHO_EQUAL,
            polynomial_order=1,
            euclidean_dimension=displacement_field.euclidean_dimension,
            basis_type=BasisType.MONOMIAL,
        )
        damage_finite_element = FiniteElement(
            element_type=ElementType.HHO_LOW,
            polynomial_order=1,
            euclidean_dimension=damage_field.euclidean_dimension,
            basis_type=BasisType.MONOMIAL,
        )

        # --- MESH
        # mesh_file_path = "meshes/plaque.msh"
        # mesh_file_path = "meshes/carre.msh"
        # mesh_file_path = "meshes/bande3.msh"
        mesh_file_path = "meshes/bande_sans_defaut.msh"
        mesh_file_path = "meshes/bande_sans_defaut2.msh"
        mmesh = Mesh(
            mesh_file_path=mesh_file_path,
            integration_order=damage_finite_element.computation_integration_order
        )

        # --- MATERIAL
        displacement_material_properties = {
            "YoungModulus": 1.0,
            "PoissonRatio": 0.0
        }
        displacement_stabilization_parameter = displacement_material_properties["YoungModulus"] / (1.0 + displacement_material_properties["PoissonRatio"])
        displacement_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            # library_path="bhv_displacement/src/libBehaviour.so",
            # library_name="PhaseFieldDisplacementDeviatoricSplit",
            library_path="bhv_displacement_spectral_split/src/libBehaviour.so",
            library_name="PhaseFieldDisplacementSpectralSplit",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=displacement_stabilization_parameter,
            lagrange_parameter=displacement_stabilization_parameter,
            field=displacement_field,
            parameters=None,
        )
        damage_material_properties = {
            "ThermalConductivity": 1.0
        }
        # ---- VALEURS MATERIAU ET CHARGEMENT OLIVIER
        # U boundary = 0.2
        # ----
        # "RegularizationLength": 0.3,
        # "FractureEnergy": 0.015
        # "YoungModulus": 210.0 (MPa)
        # "PoissonRatio": 0.3
        # ---- MAILLAGE
        # blouque sur les faces proches de la "fissure"
        # --- STABILIZATION -> axisymm
        damage_stabilization_parameter = 1000.0
        # damage_stabilization_parameter = 1.0
        damage_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            library_path="bhv_thermal/src/libBehaviour.so",
            library_name="HeatTransfer",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=damage_stabilization_parameter,
            lagrange_parameter=damage_stabilization_parameter,
            field=damage_field,
            parameters=None,
        )

        res_folder_path = get_current_res_folder_path() + "_TEMP_ANALYTICAL"

        fef_displacement = DisplacementFiniteElementField(
            mesh=mmesh,
            field=displacement_field,
            finite_element=displacement_finite_element,
            boundary_conditions=displacement_boundary_conditions,
            material=displacement_mat,
            external_variables=["Damage"],
            material_properties=displacement_material_properties,
            loads=loads,
            res_folder_path=res_folder_path,
            tolerance=1.e-6
        )

        fef_damage = DamageFiniteElementField(
            mesh=mmesh,
            field=damage_field,
            finite_element=damage_finite_element,
            boundary_conditions=damage_boundary_conditions,
            material=damage_mat,
            external_variables=["HeatSource"],
            material_properties=damage_material_properties,
            loads=[Load(volumetric_load, 0)],
            res_folder_path=res_folder_path,
            tolerance=1.e-6
        )

        cp = CoupledProblem(
            fef_displacement,
            fef_damage,
            time_steps,
            res_folder_path=res_folder_path
        )

        cp.make_time_step()
