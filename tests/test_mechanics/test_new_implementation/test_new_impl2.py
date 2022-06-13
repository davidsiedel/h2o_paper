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
    def test_new_impl(self):
        # --- VALUES
        u_min = 0.0
        u_max = 0.2
        time_steps: list = list(np.linspace(u_min, u_max, 1000, endpoint=True))
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

        def damaged(time: float, position: ndarray) -> float:
            return 10. * time

        displacement_boundary_conditions = [
            BoundaryCondition("RIGHT", pull, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("RIGHT", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("BOTTOM_DEF", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("TOP_DEF", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("TOP", fixed, BoundaryType.DISPLACEMENT, 1),
            # BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        damage_boundary_conditions = [
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("RIGHT", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("TOP_DEF", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("BOTTOM_DEF", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("TOP", fixed, BoundaryType.DISPLACEMENT, 0),
            # ---
            # BoundaryCondition("BOTTOM_DAMMAGE", fixed, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("MIDDLE", pull, BoundaryType.DISPLACEMENT, 0),
            # BoundaryCondition("TOP_DAMMAGE", fixed, BoundaryType.DISPLACEMENT, 0),
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
        mesh_file_path = "meshes/bande2.msh"
        mmesh = Mesh(
            mesh_file_path=mesh_file_path,
            integration_order=displacement_finite_element.computation_integration_order
        )

        # --- MATERIAL
        displacement_material_properties = {
            "YoungModulus": 200.0,
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
            "RegularizationLength": 0.1,
            "FractureEnergy": 1.0
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
        damage_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            library_path="bhv_damage/src/libBehaviour.so",
            library_name="PhaseFieldDamage",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=damage_stabilization_parameter,
            lagrange_parameter=damage_stabilization_parameter,
            field=damage_field,
            parameters=None,
        )

        res_folder_path = get_current_res_folder_path() + "_ROD_1000_FIXED_PT"

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
            external_variables=["HistoryFunction"],
            material_properties=damage_material_properties,
            loads=[Load(volumetric_load, 0)],
            res_folder_path=res_folder_path,
            tolerance=1.e-6
        )

        cp = CoupledProblem(
            fef_displacement,
            fef_damage,
            time_steps,
            max_iterations=1000,
            res_folder_path=res_folder_path
        )

        # cp.make_time_step()
        cp.make_time_step_fixed_point()
