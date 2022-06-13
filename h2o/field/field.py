from h2o.h2o import *
import h2o.field.fields.field_displacement as fd
import h2o.field.fields.field_scalar as fs


class Field:
    label: str
    field_type: FieldType
    flux_type: FluxType
    grad_type: GradType
    derivation_type: DerivationType
    euclidean_dimension: int
    gradient_dimension: int
    voigt_data: Dict[Tuple[int, int], Tuple[int, float]]

    def __init__(self, label: str, field_type: FieldType):
        """

        Args:
            label:
            field_type:
        """
        self.label = label
        self.field_type = field_type
        if field_type in [
            FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
            FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS,
        ]:
            self.euclidean_dimension = 2
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fd.get_plane_displacement_large_strain_data()
            )
        elif field_type in [
            FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS,
            FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
        ]:
            self.euclidean_dimension = 2
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fd.get_plane_displacement_small_strain_data()
            )
        elif field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN]:
            self.euclidean_dimension = 3
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fd.get_displacement_large_strain_data()
            )
        elif field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN]:
            self.euclidean_dimension = 3
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fd.get_displacement_small_strain_data()
            )
        elif field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN_AXISYMMETRIC]:
            self.euclidean_dimension = 2
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fd.get_axisymmetrical_displacement_large_strain_data()
            )
        elif field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN_AXISYMMETRIC]:
            self.euclidean_dimension = 2
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fd.get_axisymmetrical_displacement_small_strain_data()
            )
        elif field_type in [FieldType.SCALAR_PLANE]:
            self.euclidean_dimension = 2
            derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
                fs.get_plane_scalar_data()
            )
        else:
            raise ElementError("the specified field type is not known : {}".format(field_type))
        self.derivation_type = derivation_type
        self.flux_type = flux_type
        self.grad_type = grad_type
        self.field_dimension = field_dimension
        self.gradient_dimension = gradient_dimension
        self.voigt_data = voigt_data
        # self.label = label
        # self.field_type = field_type
        # if field_type in [
        #     FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRAIN,
        #     FieldType.DISPLACEMENT_LARGE_STRAIN_PLANE_STRESS,
        # ]:
        #     self.euclidean_dimension = 2
        #     derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
        #         fd.get_plane_displacement_large_strain_data()
        #     )
        #     self.derivation_type = derivation_type
        #     self.flux_type = flux_type
        #     self.grad_type = grad_type
        #     self.field_dimension = field_dimension
        #     self.gradient_dimension = gradient_dimension
        #     self.voigt_data = voigt_data
        # elif field_type in [
        #     FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRESS,
        #     FieldType.DISPLACEMENT_SMALL_STRAIN_PLANE_STRAIN,
        # ]:
        #     self.euclidean_dimension = 2
        #     derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
        #         fd.get_plane_displacement_small_strain_data()
        #     )
        #     self.derivation_type = derivation_type
        #     self.flux_type = flux_type
        #     self.grad_type = grad_type
        #     self.field_dimension = field_dimension
        #     self.gradient_dimension = gradient_dimension
        #     self.voigt_data = voigt_data
        # elif field_type in [FieldType.DISPLACEMENT_LARGE_STRAIN]:
        #     self.euclidean_dimension = 3
        #     derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
        #         fd.get_displacement_large_strain_data()
        #     )
        #     self.derivation_type = derivation_type
        #     self.flux_type = flux_type
        #     self.grad_type = grad_type
        #     self.field_dimension = field_dimension
        #     self.gradient_dimension = gradient_dimension
        #     self.voigt_data = voigt_data
        # elif field_type in [FieldType.DISPLACEMENT_SMALL_STRAIN]:
        #     self.euclidean_dimension = 3
        #     derivation_type, flux_type, grad_type, field_dimension, gradient_dimension, voigt_data = (
        #         fd.get_displacement_small_strain_data()
        #     )
        #     self.derivation_type = derivation_type
        #     self.flux_type = flux_type
        #     self.grad_type = grad_type
        #     self.field_dimension = field_dimension
        #     self.gradient_dimension = gradient_dimension
        #     self.voigt_data = voigt_data
