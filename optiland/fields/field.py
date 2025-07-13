"""Fields Module

This module provides classes for representing the field of view of an optical
system.

Kramer Harrison, 2023
"""


class Field:
    """Represents a field with specific properties.

    Attributes:
        field_mode (str): The type of the field.
        x (int): The x-coordinate of the field.
        y (int): The y-coordinate of the field.
        vx (float): The vignette factor in the x-direction.
        vy (float): The vignette factor in the y-direction.

    """

    def __init__(
        self,
        field_mode,
        x=0,
        y=0,
        vignette_factor_x=0.0,
        vignette_factor_y=0.0,
    ):
        self.field_mode = field_mode
        self.x = x
        self.y = y
        self.vx = vignette_factor_x
        self.vy = vignette_factor_y

    def to_dict(self):
        """Convert the field to a dictionary.

        Returns:
            dict: A dictionary representation of the field.

        """
        return {
            "field_mode": self.field_mode,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
        }

    @classmethod
    def from_dict(cls, field_dict):
        """Create a field from a dictionary.

        Args:
            field_dict (dict): A dictionary representation of the field.

        Returns:
            Field: A field object created from the dictionary.

        """
        if "field_mode" not in field_dict:
            raise ValueError("Missing required keys: field_mode")

        return cls(
            field_dict["field_mode"],
            field_dict.get("x", 0),
            field_dict.get("y", 0),
            field_dict.get("vx", 0.0),
            field_dict.get("vy", 0.0),
        )
