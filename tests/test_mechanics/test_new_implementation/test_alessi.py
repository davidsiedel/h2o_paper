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
    def test_rod_micromorphic(self):
        # --- VALUES
        u_min = 0.0
        u_max = 0.05
        time_steps: list = list(np.linspace(u_min, u_max, 100, endpoint=True))
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
            return time

        displacement_boundary_conditions = [
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("TOP", pull, BoundaryType.DISPLACEMENT, 1),
        ]

        damage_boundary_conditions = [
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 0),
            BoundaryCondition("TOP", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- FIELD
        displacement_field = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)
        damage_field = Field(label="D_CHI", field_type=FieldType.SCALAR_PLANE)

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
        mesh_file_path = "meshes/rod_alessi.msh"
        mmesh = Mesh(
            mesh_file_path=mesh_file_path,
            integration_order=displacement_finite_element.computation_integration_order
        )

        # --- MATERIAL
        displacement_material_properties = {
            "YoungModulus": 210.e3,
            "PoissonRatio": 0.3
        }
        displacement_stabilization_parameter = displacement_material_properties["YoungModulus"] / (1.0 + displacement_material_properties["PoissonRatio"])
        displacement_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            library_path="bhv_micromorphic_displacement/src/libBehaviour.so",
            library_name="MicromorphicDamageI_SpectralSplit",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=displacement_stabilization_parameter,
            lagrange_parameter=displacement_stabilization_parameter,
            field=displacement_field,
            parameters=None,
        )
        damage_material_properties = {
            "CharacteristicLength": 0.04,
            "FractureEnergy": 2.7,
            "PenalisationFactor": 300.0
        }

        damage_stabilization_parameter = 1.0
        damage_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            library_path="bhv_micromorphic_damage/src/libBehaviour.so",
            library_name="MicromorphicDamageII",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=damage_stabilization_parameter,
            lagrange_parameter=damage_stabilization_parameter,
            field=damage_field,
            parameters=None,
        )

        res_folder_path = get_current_res_folder_path() + "_ROD_ALESSI"

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
            external_variables=["EnergyReleaseRate"],
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

        cp.make_time_step_fixed_point(
            external_variable_type_d=ExternalVariable.STATE_VARIABLE,
            external_variable_type_u=ExternalVariable.STATE_VARIABLE,
        )
