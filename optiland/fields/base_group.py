"""Base Field Group Module

This module provides the abstract base class for all field groups,
which contains a collection of fields, each of the same type.
There are several field types, including:
    - angle
    - object height
    - paraxial image height
    - real image height

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod

import optiland.backend as be
from optiland.fields.field import Field


class BaseFieldGroup(ABC):
    """Base class for all field group types.

    The field group contains a collection if fields, each of the same field type.
    There are several field types, including:
        - angle
        - object height
        - paraxial image height
        - real image height

    Attributes:
        fields (list): A list of fields in the group.
        telecentric (bool): Whether the system is telecentric in object space.

    Methods:
        get_vig_factor(Hx, Hy): Returns the vignetting factors for given Hx
            and Hy values.
        get_field_coords: Returns the normalized coordinates of the fields.
        add_field(field): Adds a field to the group.
        get_field(field_number): Returns the field at the specified index.

    """

    def __init__(self):
        self.fields = []
        self.telecentric = False

    @property
    def x_fields(self):
        """be.array: x field values"""
        return be.array([field.x for field in self.fields])

    @property
    def y_fields(self):
        """be.array: y field values"""
        return be.array([field.y for field in self.fields])

    @property
    def max_x_field(self):
        """be.array: max field in x"""
        return be.max(self.x_fields)

    @property
    def max_y_field(self):
        """be.array: max field in y"""
        return be.max(self.y_fields)

    @property
    def max_field(self):
        """be.array: max field in radial coordinates"""
        return be.max(be.sqrt(self.x_fields**2 + self.y_fields**2))

    @property
    def num_fields(self):
        """int: number of fields in field group"""
        return len(self.fields)

    @property
    def vx(self):
        """be.array: vignetting factors in x"""
        return be.array([field.vx for field in self.fields])

    @property
    def vy(self):
        """be.array: vignetting factors in y"""
        return be.array([field.vy for field in self.fields])

    @abstractmethod
    def validate(self, optic):
        return  # pragma: no cover

    @abstractmethod
    def to_ray_origins(self, Hx, Hy, Px, Py, vx, vy, optic):
        return  # pragma: no cover

    @abstractmethod
    def to_paraxial_starting_ray(self, Hy, Py, wavelength, optic):
        return  # pragma: no cover

    def get_vig_factor(self, Hx, Hy):
        """Calculates the vignetting factors for a given field position.

        Note that the vignetting factors are interpolated using the nearest
        neighbor method.

        Args
            Hx (float): The normalized x component of the field.
            Hy (float): The normalized y component of the field.

        Returns:
            vx_new (float): The interpolated x-component of the
                vignetting factor.
            vy_new (float): The interpolated y-component of the
                vignetting factor.

        """
        fields = be.stack((self.x_fields, self.y_fields), axis=-1)
        v_data = be.stack((self.vx, self.vy), axis=-1)
        result = be.nearest_nd_interpolator(fields, v_data, Hx, Hy)
        vx_new = result[..., 0]
        vy_new = result[..., 1]
        return vx_new, vy_new

    def get_field_coords(self):
        """Returns the coordinates of the fields.

        If the maximum field size is 0, it returns a single coordinate (0, 0).
        Otherwise, it calculates the normalized coordinates for each field
        based on the maximum field size.

        Returns:
            list: A list of tuples representing the coordinates of the fields.

        """
        max_field = self.max_field
        if max_field == 0:
            return [(0, 0)]
        return [
            (float(x / max_field), float(y / max_field))
            for x, y in zip(self.x_fields, self.y_fields)
        ]

    def add_field(self, field):
        """Add a field to the list of fields.

        Args:
            field: The field to be added.

        """
        self.fields.append(field)

    def get_field(self, field_number):
        """Retrieve the field at the specified field_number.

        Args:
            field_number (int): The index of the field to retrieve.

        Returns:
            Field: The field at the specified index.

        Raises:
            IndexError: If the field_number is out of range.

        """
        return self.fields[field_number]

    def set_telecentric(self, is_telecentric):
        """Speocify whether the system is telecentric in object space.

        Args:
            is_telecentric (bool): Whether the system is telecentric in object
                space.

        """
        self.telecentric = is_telecentric

    def to_dict(self):
        """Convert the field group to a dictionary.

        Returns:
            dict: A dictionary representation of the field group.

        """
        return {
            "fields": [field.to_dict() for field in self.fields],
            "telecentric": self.telecentric,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a field group from a dictionary.

        Args:
            data (dict): A dictionary representation of the field group.

        Returns:
            FieldGroup: A field group object created from the dictionary.

        """
        field_group = cls()
        for field_dict in data["fields"]:
            field_group.add_field(Field.from_dict(field_dict))
        field_group.set_telecentric(data["telecentric"])
        return field_group
