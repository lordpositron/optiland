"""Angle Field Group Module.

This module defines the `AngleFieldGroup` class, which represents a collection
of fields defined by angles relative to the optical axis.
"""

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.fields.base_group import BaseFieldGroup

if TYPE_CHECKING:
    from optiland.optic.optic import Optic  # pragma: no cover


class AngleFieldGroup(BaseFieldGroup):
    """Represents a group of fields defined by angles.

    This field group type is used when the field points are specified as angles
    (typically in degrees) with respect to the optical axis. It validates that
    the system is not telecentric in object space, as angle fields are
    incompatible with object-space telecentricity.

    The actual conversion of these angles into ray starting conditions is
    handled by `to_ray_origins` and `to_paraxial_starting_ray`, which need
    concrete implementations based on the optical system's conventions (e.g.,
    how angles translate to ray vectors at the object surface or first surface).
    """

    def __init__(self) -> None:
        """Initializes a new `AngleFieldGroup` instance."""
        super().__init__()

    @classmethod
    def field_type_string(cls) -> str:
        """Returns the string identifier for this field group type.

        Returns:
            str: The string "angle".
        """
        return "angle"

    def validate(self, optic: "Optic") -> None:
        """Validates the angle field group configuration against the optic.

        Ensures that the `optic.field_type` (if it exists and is checked)
        is compatible and that the system is not object-space telecentric.

        Args:
            optic (Optic): The optical system instance.

        Raises:
            ValueError: If `optic.obj_space_telecentric` is `True`, as angle
                fields are not compatible with telecentric object space.
        """
        if optic.obj_space_telecentric:
            raise ValueError(
                f"Field type '{self.field_type_string()}' cannot be used when "
                "object_space_telecentric is True."
            )
        self.telecentric = (
            optic.obj_space_telecentric
        )  # Keep its own telecentric flag in sync

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
        """Converts normalized angle field and pupil coords to ray origins/directions.

        This method must be implemented to define how angle fields (represented by
        `self.fields` and normalized `Hx`, `Hy`) translate into ray starting
        positions and direction cosines at the object surface or entrance pupil.

        Args:
            Hx (be.ndarray): Normalized x-coordinates of the field angles.
            Hy (be.ndarray): Normalized y-coordinates of the field angles.
            Px (be.ndarray): Normalized x-coordinates of the pupil points.
            Py (be.ndarray): Normalized y-coordinates of the pupil points.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.
            optic (Optic): The optical system.

        Returns:
            tuple[be.ndarray, be.ndarray]: Ray origins (ro) and directions (rd).

        Raises:
            NotImplementedError: This method is abstract and must be implemented
                by a concrete subclass or specialization.
        """
        # Example of what might be needed:
        # field_x_angles = Hx * self.max_x_field
        # field_y_angles = Hy * self.max_y_field
        # ... logic to convert angles to direction cosines ...
        # ... logic to determine ray origin (e.g., on object surface) ...
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_ray_origins is not yet implemented."
        )  # pragma: no cover

    def to_paraxial_starting_ray(
        self, Hy: float, Py: float, wavelength: float, optic: "Optic"
    ) -> tuple[float, float]:
        """Calculates starting height and angle for a paraxial ray from an angle field.

        This method must be implemented to define how a normalized field angle
        (`Hy` scaled by `self.max_y_field`) translates into an initial height (y)
        and angle (u) for paraxial tracing.

        Args:
            Hy (float): Normalized y-coordinate of the field angle.
            Py (float): Normalized y-coordinate of the pupil point.
            wavelength (float): The wavelength of the ray in microns.
            optic (Optic): The optical system.

        Returns:
            tuple[float, float]: Initial height (y) and angle (u).

        Raises:
            NotImplementedError: This method is abstract and must be implemented
                by a concrete subclass or specialization.
        """
        # Example of what might be needed:
        # field_y_angle_rad = be.deg2rad(Hy * self.max_y_field)
        # initial_y = 0.0 # If object at infinity, ray starts at axis
        # initial_u = be.tan(field_y_angle_rad)
        # # Or just field_y_angle_rad for small angles
        raise NotImplementedError(
            f"{self.__class__.__name__}.to_paraxial_starting_ray is not "
            "yet implemented."
        )  # pragma: no cover

    def set_telecentric(self, is_telecentric: bool) -> None:
        """Sets the object space telecentricity flag.

        For `AngleFieldGroup`, if `is_telecentric` is `True`, this method
        will raise a `ValueError` because angle fields are fundamentally
        incompatible with object-space telecentric systems (where the chief
        rays are parallel to the optical axis in object space, implying an
        infinite entrance pupil distance, not an angle definition).

        Args:
            is_telecentric (bool): Whether the system is intended to be
                telecentric in object space.

        Raises:
            ValueError: If `is_telecentric` is `True`.
        """
        if is_telecentric:
            raise ValueError(
                f"Field type '{self.field_type_string()}' cannot be used when "
                "object_space_telecentric is True. "
                "Cannot set AngleFieldGroup to be telecentric."
            )
        super().set_telecentric(is_telecentric)  # Should be False
