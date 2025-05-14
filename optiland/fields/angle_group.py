from optiland.fields.base_group import BaseFieldGroup


class AngleFieldGroup(BaseFieldGroup):
    def __init__(self):
        super().__init__()

    def validate(self, optic):
        if optic.field_type != "angle":
            raise ValueError("Incompatible field types. Must be of type 'angle'.")

        if optic.obj_space_telecentric:
            raise ValueError(
                'Field type cannot be "angle" for telecentric object space.'
            )

    def to_ray_origins(self, Hx, Hy, Px, Py, vx, vy, optic):
        return  # pragma: no cover

    def to_paraxial_starting_ray(self, Hy, Py, wavelength, optic):
        return  # pragma: no cover

    def set_telecentric(self, is_telecentric):
        """Specify whether the system is telecentric in object space.

        Args:
            is_telecentric (bool): Whether the system is telecentric in object
                space.

        """
        raise ValueError('Field type cannot be "angle" for telecentric object space.')
