from h2o.h2o import *
from h2o.fem.basis.basis import Basis


class FiniteElement:
    element_type: ElementType
    k_order: int
    construction_integration_order: int
    computation_integration_order: int
    basis_type: BasisType
    cell_basis_k: Basis
    cell_basis_l: Basis
    face_basis_k: Basis
    cell_basis_r: Basis

    def __init__(
        self,
        element_type: ElementType,
        polynomial_order: int,
        euclidean_dimension: int,
        basis_type: BasisType = BasisType.MONOMIAL,
    ):
        """

        Args:
            element_type:
            polynomial_order:
            euclidean_dimension:
            basis_type:
        """
        self.element_type = element_type
        self.basis_type = basis_type
        # --- POLYNOMIAL ORDERS
        self.k_order = polynomial_order
        self.r_order = polynomial_order + 1
        if element_type in [ElementType.HDG_LOW, ElementType.HHO_LOW]:
            self.l_order = polynomial_order - 1
        elif element_type in [ElementType.HDG_EQUAL, ElementType.HHO_EQUAL]:
            self.l_order = polynomial_order
        elif element_type in [ElementType.HDG_HIGH, ElementType.HHO_HIGH]:
            self.l_order = polynomial_order + 1
        else:
            raise ElementError("the specified element type is not known : {}".format(element_type))
        # --- BUILDING BASES
        self.cell_basis_k = Basis(self.k_order, euclidean_dimension, basis_type=basis_type)
        self.cell_basis_l = Basis(self.l_order, euclidean_dimension, basis_type=basis_type)
        self.cell_basis_r = Basis(self.r_order, euclidean_dimension, basis_type=basis_type)
        self.face_basis_k = Basis(self.k_order, euclidean_dimension - 1, basis_type=basis_type)
        # --- INTEGRATION ORDERS
        # self.construction_integration_order = 2 * (polynomial_order + 1)
        self.construction_integration_order = 2 * self.cell_basis_r.polynomial_order + 1
        self.computation_integration_order = 2 * self.cell_basis_k.polynomial_order + 1
        # self.construction_integration_order = 3 * self.cell_basis_r.polynomial_order
        # self.computation_integration_order = 3 * self.cell_basis_k.polynomial_order
        # self.computation_integration_order = 2 * self.construction_integration_order
        # ---- CARTESIAN
        self.construction_integration_order = 2 * self.cell_basis_r.polynomial_order
        self.computation_integration_order = 2 * self.cell_basis_k.polynomial_order
        print("----------------------------------------------------------------------------------------------------")
        print("+ FACE BASIS K ORDER : {} | FACE BASIS K DIMENSION : {}".format(self.face_basis_k.polynomial_order, self.face_basis_k.dimension))
        print("+ CELL BASIS K ORDER : {} | CELL BASIS K DIMENSION : {}".format(self.cell_basis_k.polynomial_order, self.cell_basis_k.dimension))
        print("+ CELL BASIS R ORDER : {} | CELL BASIS R DIMENSION : {}".format(self.cell_basis_r.polynomial_order, self.cell_basis_r.dimension))
        print("+ CELL BASIS L ORDER : {} | CELL BASIS L DIMENSION : {}".format(self.cell_basis_l.polynomial_order, self.cell_basis_l.dimension))
        print("+ COMPUTATION INTEGRATION ORDER : {}".format(self.computation_integration_order))
        print("----------------------------------------------------------------------------------------------------")
        # self.computation_integration_order = 8
        # self.construction_integration_order = 2 * (polynomial_order)
        # self.computation_integration_order = 2 * (polynomial_order)