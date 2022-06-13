from unittest import TestCase

import numpy as np

from h2o.problem.finite_element_field import FiniteElementField
from h2o.problem.displacement_finite_element_field import DisplacementFiniteElementField
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
        u_max = 0.1
        u_max = 1.e-3
        time_steps: list = list(np.linspace(u_min, u_max, 4, endpoint=True))
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
            BoundaryCondition("BOTTOM", fixed, BoundaryType.DISPLACEMENT, 1),
            BoundaryCondition("LEFT", fixed, BoundaryType.DISPLACEMENT, 0),
        ]

        # --- MESH
        mesh_file_path = "meshes/ssna_quad_light.msh"

        # --- FIELD
        displacement = Field(label="U", field_type=FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN)

        damage = Field(label="D", field_type=FieldType.SCALAR_PLANE)

        # --- FINITE ELEMENT
        displacement_finite_element = FiniteElement(
            element_type=ElementType.HHO_HIGH,
            polynomial_order=1,
            euclidean_dimension=displacement.euclidean_dimension,
            basis_type=BasisType.MONOMIAL,
        )

        # --- FINITE ELEMENT
        damage_finite_element = FiniteElement(
            element_type=ElementType.HHO_LOW,
            polynomial_order=1,
            euclidean_dimension=damage.euclidean_dimension,
            basis_type=BasisType.MONOMIAL,
        )

        mmesh = Mesh(
            mesh_file_path=mesh_file_path,
            integration_order=displacement_finite_element.computation_integration_order
        )

        # --- MATERIAL

        damage_parameters = {
            "RegularizationLength": 0.02,
            "FractureEnergy": 1.
        }

        displacement_material_properties = {"YoungModulus": 200.0e9, "PoissonRatio": 0.3}

        stabilization_parameter = displacement_material_properties["YoungModulus"] / (1.0 + displacement_material_properties["PoissonRatio"])

        displacement_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            library_path="bhv_displacement/src/libBehaviour.so",
            library_name="PhaseFieldDisplacementDeviatoricSplit",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=stabilization_parameter,
            field=displacement,
            parameters=None,
        )

        damage_mat = Material(
            nq=mmesh.number_of_cell_quadrature_points_in_mesh,
            library_path="bhv_damage/src/libBehaviour.so",
            library_name="PhaseFieldDamage",
            hypothesis=mgis_bv.Hypothesis.PLANESTRAIN,
            stabilization_parameter=stabilization_parameter,
            lagrange_parameter=1.0,
            field=damage,
            parameters=None,
        )

        print(damage_mat.mat_data.s0.gradients[0])
        mgis_bv.setMaterialProperty(damage_mat.mat_data.s0, 'RegularizationLength', 0.04)
        mgis_bv.setMaterialProperty(damage_mat.mat_data.s1, 'RegularizationLength', 0.04)
        mgis_bv.setMaterialProperty(damage_mat.mat_data.s0, 'FractureEnergy', 1)
        mgis_bv.setMaterialProperty(damage_mat.mat_data.s1, 'FractureEnergy', 1)
        mgis_bv.setExternalStateVariable(damage_mat.mat_data.s0, 'Temperature', 200)
        mgis_bv.setExternalStateVariable(damage_mat.mat_data.s1, 'Temperature', 200)
        mgis_bv.setExternalStateVariable(damage_mat.mat_data.s0, 'HistoryFunction', np.zeros((damage_mat.nq,)), mgis_bv.MaterialStateManagerStorageMode.LOCAL_STORAGE)
        mgis_bv.setExternalStateVariable(damage_mat.mat_data.s1, 'HistoryFunction', np.zeros((damage_mat.nq,)), mgis_bv.MaterialStateManagerStorageMode.LOCAL_STORAGE)
        # mgis_bv.setExternalStateVariable(damage_mat.mat_data.s0, 'HistoryFunction', 200)
        # mgis_bv.setExternalStateVariable(damage_mat.mat_data.s1, 'HistoryFunction', 200)
        integ_res = mgis_bv.integrate(damage_mat.mat_data, damage_mat.integration_type, 0, 0, (0 + 1))
        # print(damage_mat.mat_data.K)
        # print(damage_mat.mat_data.K[0].size)

        res_folder_path = get_current_res_folder_path() + "_NEW_IMPL2"

        fef = FiniteElementField(
            mesh=mmesh,
            field=damage,
            finite_element=damage_finite_element,
            boundary_conditions=boundary_conditions,
            material=damage_mat,
            external_variables=["HistoryFunction"],
            material_properties={"FractureEnergy": 1.0, "RegularizationLength": 0.04},
            loads=loads,
            res_folder_path=res_folder_path,
            tolerance=1.e-6
        )

        fef_displacement = DisplacementFiniteElementField(
            mesh=mmesh,
            field=displacement,
            finite_element=displacement_finite_element,
            boundary_conditions=boundary_conditions,
            material=displacement_mat,
            external_variables=["Damage"],
            material_properties=displacement_material_properties,
            loads=loads,
            res_folder_path=res_folder_path,
            tolerance=1.e-6
        )

        cp = CoupledProblem(
            [fef_displacement],
            time_steps,
            res_folder_path=res_folder_path
        )

        cp.make_time_step()
