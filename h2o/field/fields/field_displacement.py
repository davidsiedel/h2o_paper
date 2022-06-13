from h2o.h2o import *


def get_plane_displacement_large_strain_data() -> (
    DerivationType,
    FluxType,
    GradType,
    int,
    int,
    Dict[Tuple[int, int], Tuple[int, float]],
):
    """

    :return:
    """
    derivation_type = DerivationType.REGULAR
    flux_type = FluxType.STRESS_PK1
    grad_type = GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT
    field_dimension = 2
    gradient_dimension = 5
    voigt_data = {
        (0, 0): (0, 1.0),
        (1, 1): (1, 1.0),
        (0, 1): (3, 1.0),
        (1, 0): (4, 1.0),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data

def get_axisymmetrical_displacement_large_strain_data() -> (
    DerivationType,
    FluxType,
    GradType,
    int,
    int,
    Dict[Tuple[int, int], Tuple[int, float]],
):
    """

    :return:
    """
    derivation_type = DerivationType.LARGE_STRAIN_AXISYMMETRIC
    flux_type = FluxType.STRESS_PK1
    grad_type = GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT
    field_dimension = 2
    gradient_dimension = 5
    voigt_data = {
        (0, 0): (0, 1.0),
        # (2, 2): (1, 1.0),
        # (1, 1): (2, 1.0),
        (2, 2): (2, 1.0),
        (1, 1): (1, 1.0),
        (0, 1): (3, 1.0),
        (1, 0): (4, 1.0),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data


def get_plane_displacement_small_strain_data() -> (
    DerivationType,
    FluxType,
    GradType,
    int,
    int,
    Dict[Tuple[int, int], Tuple[int, float]],
):
    """

    :return:
    """
    derivation_type = DerivationType.SYMMETRIC
    flux_type = FluxType.STRESS_CAUCHY
    grad_type = GradType.DISPLACEMENT_SMALL_STRAIN
    field_dimension = 2
    gradient_dimension = 4
    voigt_data = {
        (0, 0): (0, 1.0),
        (1, 1): (1, 1.0),
        (0, 1): (3, np.sqrt(2.0)),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data


def get_axisymmetrical_displacement_small_strain_data() -> (
    DerivationType,
    FluxType,
    GradType,
    int,
    int,
    Dict[Tuple[int, int], Tuple[int, float]],
):
    """

    :return:
    """
    derivation_type = DerivationType.SMALL_STRAIN_AXISYMMETRIC
    flux_type = FluxType.STRESS_CAUCHY
    grad_type = GradType.DISPLACEMENT_SMALL_STRAIN
    field_dimension = 2
    gradient_dimension = 4
    voigt_data = {
        (0, 0): (0, 1.0),
        # (2, 2): (1, 1.0),
        # (1, 1): (2, 1.0),
        (2, 2): (2, 1.0),
        (1, 1): (1, 1.0),
        (0, 1): (3, np.sqrt(2.0)),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data


def get_displacement_large_strain_data() -> (
    DerivationType,
    FluxType,
    GradType,
    int,
    int,
    Dict[Tuple[int, int], Tuple[int, float]],
):
    """

    :return:
    """
    derivation_type = DerivationType.REGULAR
    flux_type = FluxType.STRESS_PK1
    grad_type = GradType.DISPLACEMENT_TRANSFORMATION_GRADIENT
    field_dimension = 3
    gradient_dimension = 9
    voigt_data = {
        (0, 0): (0, 1.0),
        (1, 1): (1, 1.0),
        (2, 2): (2, 1.0),
        (0, 1): (3, 1.0),
        (1, 0): (4, 1.0),
        (0, 2): (5, 1.0),
        (2, 0): (6, 1.0),
        (1, 2): (7, 1.0),
        (2, 1): (8, 1.0),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data


def get_displacement_small_strain_data() -> (
    DerivationType,
    FluxType,
    GradType,
    int,
    int,
    Dict[Tuple[int, int], Tuple[int, float]],
):
    """

    :return:
    """
    derivation_type = DerivationType.SYMMETRIC
    flux_type = FluxType.STRESS_CAUCHY
    grad_type = GradType.DISPLACEMENT_SMALL_STRAIN
    field_dimension = 3
    gradient_dimension = 6
    voigt_data = {
        (0, 0): (0, 1.0),
        (1, 1): (1, 1.0),
        (2, 2): (2, 1.0),
        (0, 1): (3, np.sqrt(2.0)),
        (0, 2): (4, np.sqrt(2.0)),
        (1, 2): (5, np.sqrt(2.0)),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data
