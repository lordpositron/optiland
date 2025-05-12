from optiland.fields.base_group import BaseFieldGroup


class AngleFieldGroup(BaseFieldGroup):
    def __init__(self):
        super().__init__()

    def validate(self, optic):
        return

    def to_ray_origins(self, Hx, Hy, Px, Py, vx, vy, optic):
        return

    def to_paraxial_starting_ray(self, Hy, Py, wavelength, optic):
        return

    def set_telecentric(self, is_telecentric):
        """Specify whether the system is telecentric in object space.

        Args:
            is_telecentric (bool): Whether the system is telecentric in object
                space.

        """
        raise ValueError('Field type cannot be "angle" for telecentric object space.')
