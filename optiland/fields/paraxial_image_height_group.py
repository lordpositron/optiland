"""Paraxial Image Height Field Group Module.

This module defines the `ParaxialImageHeightGroup` class, which represents
a collection of field points specified by their heights in the paraxial
image plane.
"""

from typing import TYPE_CHECKING  # Tuple removed

import optiland.backend as be
from optiland.fields.base_group import BaseFieldGroup

if TYPE_CHECKING:
    from optiland.optic.optic import Optic  # pragma: no cover


class ParaxialImageHeightGroup(BaseFieldGroup):
    """A field group where fields are defined by paraxial image heights.

    The `x` and `y` attributes of `Field` objects within this group represent
    heights in the paraxial image plane. This group provides methods to convert
    these paraxial image heights into object-space ray parameters for both
    paraxial and real ray tracing.

    Attributes:
        fields (List[Field]): A list of `Field` objects. Each field's `x` and `y`
            are interpreted as paraxial image heights.
    """

    def __init__(self) -> None:
        """Initializes a new `ParaxialImageHeightGroup` instance."""
        super().__init__()

    @classmethod
    def field_type_string(cls) -> str:
        """Returns the string identifier for this field group type.

        Returns:
            str: The type string "paraxial_image_height".
        """
        return "paraxial_image_height"

    def validate(self, optic: "Optic") -> None:
        """Validates the field group configuration against the optic.

        Checks if the optical system can form a finite paraxial image.
        For example, afocal systems might be incompatible if paraxial image
        height is ill-defined.

        Args:
            optic (Optic): The optical system instance to validate against.

        Raises:
            ValueError: If the optic configuration is incompatible with
                        paraxial image height fields.
        """
        try:
            efl = optic.paraxial.f2()
            mag = optic.paraxial.magnification()
        except Exception as e:
            raise ValueError(
                "Could not retrieve paraxial properties (EFL, Magnification) "
                f"for validation. System may be ill-defined. Original error: {e}"
            ) from e

        if optic.object_surface and optic.object_surface.is_infinite:
            if be.isinf(efl) or be.isclose(efl, 0.0):
                # For an object at infinity, an infinite or zero EFL implies
                # an afocal system or one where image height isn't well-defined.
                # (e.g. telescope forming image at infinity, angular mag is key)
                # Allow for now but could be stricter.
                pass
        elif optic.object_surface:  # Finite object
            if be.isclose(mag, 0.0):
                raise ValueError(
                    "Paraxial magnification is zero for a finite object, "
                    "implying image at infinity. "
                    "ParaxialImageHeightGroup is not suitable."
                )
            if be.isinf(mag):
                raise ValueError(
                    "Paraxial magnification is infinite for a finite object "
                    "(object likely at front focal point), implying image at "
                    "infinity. ParaxialImageHeightGroup is not suitable."
                )
        else:  # No object surface
            raise ValueError("Optic requires an object surface for this validation.")

    def to_paraxial_starting_ray(
        self, Hy: float, Py: float, wavelength: float, optic: "Optic"
    ) -> tuple[float, float]:
        """Calculates starting object-space height and angle for a paraxial ray.

        Args:
            Hy (float): Normalized y-coordinate of the paraxial image height.
            Py (float): Normalized y-coordinate of the pupil point.
            wavelength (float): The wavelength of the ray in microns.
            optic (Optic): The optical system.

        Returns:
            Tuple[float, float]: (y_start, u_start) at object/first surface.
        """
        y_image_target = Hy * self.max_y_field

        try:
            mag = optic.paraxial.magnification()
            efl = optic.paraxial.f2()
            epd = optic.paraxial.EPD()
            epl_z = optic.paraxial.EPL()
        except Exception as e:
            raise ValueError(
                "Failed to get paraxial properties from optic "
                f"(mag, efl, epd, epl_z). Original error: {e}"
            ) from e

        if optic.object_surface and optic.object_surface.is_infinite:
            if be.isclose(efl, 0.0):
                raise ValueError(
                    "Effective focal length is zero for object at infinity."
                )

            theta_object_angle_tan = y_image_target / efl
            u_start = theta_object_angle_tan

            if not optic.surface_group.surfaces:
                raise ValueError("Optic has no surfaces for infinite object ray start.")
            first_surface_z = optic.surface_group.surfaces[0].geometry.cs.z

            y_pupil_offset = Py * (epd / 2.0)
            y_start = y_pupil_offset + u_start * (first_surface_z - epl_z)

        elif optic.object_surface:  # Finite object distance
            if be.isclose(mag, 0.0) or be.isinf(mag):
                raise ValueError(
                    f"Paraxial magnification is {mag} for finite object. "
                    "Cannot determine object height from image height reliably."
                )
            y_object_field = y_image_target / mag
            y_start = y_object_field

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
        else:
            raise ValueError("Optic has no object surface defined.")

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
        """Converts norm. image heights & pupil coords to ray origins & directions.

        Args:
            Hx: Normalized x paraxial image heights.
            Hy: Normalized y paraxial image heights.
            Px: Normalized x pupil coordinates.
            Py: Normalized y pupil coordinates.
            vx: Vignetting factor in x.
            vy: Vignetting factor in y.
            optic: The optical system.

        Returns:
            Tuple[be.ndarray, be.ndarray]: Ray origins (ro) and directions (rd).
        """
        x_image_target = Hx * self.max_x_field
        y_image_target = Hy * self.max_y_field

        try:
            mag = optic.paraxial.magnification()
            efl = optic.paraxial.f2()
            epd = optic.paraxial.EPD()
            epl_z = optic.paraxial.EPL()
        except Exception as e:
            raise ValueError(
                "Failed to get paraxial properties for to_ray_origins. "
                f"Original error: {e}"
            ) from e

        if optic.object_surface and optic.object_surface.is_infinite:
            if be.isclose(efl, 0.0):
                raise ValueError(
                    "Effective focal length is zero for object at infinity."
                )

            theta_x_tan = x_image_target / efl
            theta_y_tan = y_image_target / efl

            if not optic.surface_group.surfaces:
                raise ValueError("Optic has no surfaces to define ray origins.")

            first_surface_z = optic.surface_group.surfaces[0].geometry.cs.z

            ro_x = Px * (epd / 2.0)
            ro_y = Py * (epd / 2.0)
            ro = be.stack([ro_x, ro_y, be.full_like(ro_x, first_surface_z)], axis=-1)

            # Using l,m,n for direction cosines is standard.
            l_cos = be.sin(be.atan(theta_x_tan))  # noqa: E741
            m_cos = be.sin(be.atan(theta_y_tan))
            n_squared = 1.0 - l_cos**2 - m_cos**2
            n_squared = be.where(n_squared < 0, 0, n_squared)
            n_cos = be.sqrt(n_squared)
            rd = be.normalize(be.stack([l_cos, m_cos, n_cos], axis=-1))

        elif optic.object_surface:  # Finite object distance
            if be.isclose(mag, 0.0) or be.isinf(mag):
                raise ValueError(
                    f"Paraxial magnification is {mag} for finite object. "
                    "Cannot determine object positions reliably."
                )

            x_object_field = x_image_target / mag
            y_object_field = y_image_target / mag

            z_object = optic.object_surface.geometry.cs.z
            ro = be.stack(
                [
                    x_object_field,
                    y_object_field,
                    be.full_like(x_object_field, z_object),
                ],
                axis=-1,
            )

            pupil_point_x = Px * (epd / 2.0)
            pupil_point_y = Py * (epd / 2.0)

            vec_x = pupil_point_x - x_object_field
            vec_y = pupil_point_y - y_object_field
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
        else:
            raise ValueError("Optic has no object surface defined.")

        return ro, rd
