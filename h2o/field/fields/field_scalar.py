from h2o.h2o import *


def get_plane_scalar_data() -> (
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
    flux_type = FluxType.STRESS_CAUCHY
    grad_type = GradType.DISPLACEMENT_SMALL_STRAIN
    field_dimension = 1
    gradient_dimension = 2
    voigt_data = {
        (0, 0): (0, 1.0),
        (0, 1): (1, 1.0),
    }
    return derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data
