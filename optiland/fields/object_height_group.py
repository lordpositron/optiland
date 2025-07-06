"""Object Height Field Group Module.

This module defines the `ObjectHeightGroup` class, which represents
a collection of field points specified by their heights in the object plane.
"""

from typing import TYPE_CHECKING  # Tuple removed

import optiland.backend as be
from optiland.fields.base_group import BaseFieldGroup

if TYPE_CHECKING:
    from optiland.optic.optic import Optic  # pragma: no cover


class ObjectHeightGroup(BaseFieldGroup):
    """A field group where fields are defined by object heights.

    The `x` and `y` attributes of `Field` objects within this group represent
    heights on the object surface. This group provides methods to convert
    these object heights into object-space ray parameters for both
    paraxial and real ray tracing.

    This field group is only applicable if the object surface is at a finite
    distance.

    Attributes:
        fields (List[Field]): A list of `Field` objects. Each field's `x` and `y`
            are interpreted as object heights.
    """

    def __init__(self) -> None:
        """Initializes a new `ObjectHeightGroup` instance."""
        super().__init__()

    @classmethod
    def field_type_string(cls) -> str:
        """Returns the string identifier for this field group type.

        Returns:
            str: The type string "object_height".
        """
        return "object_height"

    def validate(self, optic: "Optic") -> None:
        """Validates the field group configuration against the optic.

        Checks if the optical system's object surface is at a finite distance.
        Object height fields are not meaningful for objects at infinity.

        Args:
            optic (Optic): The optical system instance to validate against.

        Raises:
            ValueError: If the optic's object surface is at infinity or not defined.
        """
        if not optic.object_surface:
            raise ValueError(
                "ObjectHeightGroup requires an object surface to be defined."
            )
        if optic.object_surface.is_infinite:
            raise ValueError(
                "ObjectHeightGroup cannot be used when object surface is at infinity."
            )

    def to_paraxial_starting_ray(
        self, Hy: float, Py: float, wavelength: float, optic: "Optic"
    ) -> tuple[float, float]:
        """Calculates starting object-space height and angle for a paraxial ray.

        Args:
            Hy (float): Normalized y-coordinate of the object height.
            Py (float): Normalized y-coordinate of the pupil point.
            wavelength (float): The wavelength of the ray in microns.
            optic (Optic): The optical system.

        Returns:
            Tuple[float, float]: (y_start, u_start) at the object surface.
        """
        if not optic.object_surface or optic.object_surface.is_infinite:
            raise ValueError("ObjectHeightGroup requires a finite object surface.")

        y_start = Hy * self.max_y_field

        try:
            epd = optic.paraxial.EPD()
            epl_z = optic.paraxial.EPL()
        except Exception as e:
            raise ValueError(
                "Failed to get paraxial properties (EPD, EPL_z) from optic. "
                f"Original error: {e}"
            ) from e

        z_object = optic.object_surface.geometry.cs.z
        y_pupil_height_in_epp = Py * (epd / 2.0)

        delta_z_pupil_obj = epl_z - z_object
        if be.isclose(delta_z_pupil_obj, 0.0):
            if optic.obj_space_telecentric:
                u_start = 0.0
            else:
                if not be.isclose(Py, 0.0) and not be.isclose(
                    y_pupil_height_in_epp, y_start
                ):
                    raise ValueError(
                        "Object plane at EPL for non-telecentric system "
                        "leads to ambiguous ray angle."
                    )
                u_start = 0.0
        else:
            u_start = (y_pupil_height_in_epp - y_start) / delta_z_pupil_obj

        return float(y_start), float(u_start)

    def to_ray_origins(
        self,
        Hx: be.ndarray,
        Hy: be.ndarray,
        Px: be.ndarray,
        Py: be.ndarray,
        vx: float,
        vy: float,
        optic: "Optic",
    ) -> tuple[be.ndarray, be.ndarray]:
        """Converts norm. object heights & pupil coords to ray origins & directions.

        Args:
            Hx: Normalized x object heights.
            Hy: Normalized y object heights.
            Px: Normalized x pupil coordinates.
            Py: Normalized y pupil coordinates.
            vx: Vignetting factor in x.
            vy: Vignetting factor in y.
            optic: The optical system.

        Returns:
            Tuple[be.ndarray, be.ndarray]: Ray origins (ro) and directions (rd).
        """
        if not optic.object_surface or optic.object_surface.is_infinite:
            raise ValueError("ObjectHeightGroup requires a finite object surface.")

        x_object_actual = Hx * self.max_x_field
        y_object_actual = Hy * self.max_y_field

        z_object = optic.object_surface.geometry.cs.z
        ro = be.stack(
            [x_object_actual, y_object_actual, be.full_like(x_object_actual, z_object)],
            axis=-1,
        )

        try:
            epd = optic.paraxial.EPD()
            epl_z = optic.paraxial.EPL()
        except Exception as e:
            raise ValueError(
                "Failed to get paraxial properties (EPD, EPL_z) from optic. "
                f"Original error: {e}"
            ) from e

        pupil_point_x = Px * (epd / 2.0)
        pupil_point_y = Py * (epd / 2.0)

        vec_x = pupil_point_x - x_object_actual
        vec_y = pupil_point_y - y_object_actual
        vec_z_scalar = epl_z - z_object

        if be.isclose(vec_z_scalar, 0.0) and optic.obj_space_telecentric:
            rd = be.zeros_like(ro)
            rd[..., 2] = 1.0
        else:
            direction_vectors = be.stack(
                [vec_x, vec_y, be.full_like(vec_x, vec_z_scalar)], axis=-1
            )
            rd = be.normalize(direction_vectors)

            if be.any(be.all(be.isclose(direction_vectors, 0.0), axis=-1)):
                is_zero_vec = be.all(be.isclose(direction_vectors, 0.0), axis=-1)
                default_rd = be.zeros_like(rd)
                default_rd[..., 2] = 1.0
                rd = be.where(is_zero_vec[..., be.newaxis], default_rd, rd)

        return ro, rd
