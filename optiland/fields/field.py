"""Field Module.

This module defines the `Field` class, which represents a single field point in
an optical system. A field point is typically defined by its coordinates (x, y)
and associated vignetting factors.
"""

from typing import Any


class Field:
    """Represents a single field point in an optical system.

    A field point defines a specific point in the object or image plane, or an
    angle of incidence, for which rays are traced. It typically includes spatial
    coordinates and vignetting factors. The interpretation of the coordinates (e.g.,
    as angles or heights) depends on the `FieldGroup` that manages this field.

    Attributes:
        x (float): The x-coordinate or x-component of the field. Its physical
            meaning (e.g., object height in mm, angle in degrees) is determined
            by the context of the `FieldGroup` it belongs to. Defaults to 0.0.
        y (float): The y-coordinate or y-component of the field. Its physical
            meaning is determined by the context of the `FieldGroup`. Defaults to 0.0.
        vx (float): The vignetting factor in the x-direction for this field.
            This factor typically scales the pupil extent for rays from this
            field point. Defaults to 0.0 (no vignetting).
        vy (float): The vignetting factor in the y-direction for this field.
            Defaults to 0.0 (no vignetting).
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        vignette_factor_x: float = 0.0,
        vignette_factor_y: float = 0.0,
    ) -> None:
        """Initializes a new `Field` instance.

        Args:
            x (float, optional): The x-coordinate or component of the field.
                Defaults to 0.0.
            y (float, optional): The y-coordinate or component of the field.
                Defaults to 0.0.
            vignette_factor_x (float, optional): The vignetting factor in the
                x-direction. Defaults to 0.0.
            vignette_factor_y (float, optional): The vignetting factor in the
                y-direction. Defaults to 0.0.
        """
        self.x: float = x
        self.y: float = y
        self.vx: float = vignette_factor_x
        self.vy: float = vignette_factor_y

    def to_dict(self) -> dict[str, float]:
        """Serializes the field attributes to a dictionary.

        The `field_type` is no longer part of the `Field` class itself, as it's
        managed by the `FieldGroup`.

        Returns:
            dict[str, float]: A dictionary containing the 'x', 'y', 'vx', and 'vy'
            attributes of the field.
        """
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Field":
        """Creates a `Field` instance from a dictionary representation.

        The dictionary is expected to contain keys 'x', 'y', 'vx', and 'vy'.
        If a key is missing, its corresponding attribute will be set to a
        default value (0 for coordinates, 0.0 for vignetting factors).

        Args:
            data (dict[str, Any]): A dictionary containing the field's attributes.
                Expected keys: "x", "y", "vx", "vy".

        Returns:
            Field: A new `Field` object populated from the dictionary data.
        """
        # Note: The original `from_dict` checked for "field_type".
        # This is removed as field_type is no longer an attribute of Field.
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            vignette_factor_x=float(data.get("vx", 0.0)),
            vignette_factor_y=float(data.get("vy", 0.0)),
        )

    def __repr__(self) -> str:
        """Provides a string representation of the Field instance."""
        return f"Field(x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy})"
