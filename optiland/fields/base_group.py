"""Base Field Group Module.

This module provides the abstract base class for all field groups. A field
group contains a collection of fields, all of which must be of the same
conceptual type (e.g., all angle fields, all object height fields).

The primary responsibilities of a `BaseFieldGroup` derivative are:
    - Managing a list of `Field` objects.
    - Providing methods to convert field specifications into ray data suitable for
      tracing (e.g., `to_ray_origins`, `to_paraxial_starting_ray`).
    - Validating the compatibility of its field type with global optic settings
      (e.g., telecentricity).
    - Serialization and deserialization of its state.

Typical field types include:
    - Angle: Fields defined by an angle with respect to the optical axis.
    - Object Height: Fields defined by a height on the object plane.
    - Paraxial Image Height: Fields defined by a height in the paraxial image
      plane.
    - Real Image Height: Fields defined by a height in the real image plane.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.fields.field import Field

if TYPE_CHECKING:
    from optiland.optic.optic import Optic  # pragma: no cover


class BaseFieldGroup(ABC):
    """Abstract base class for all field group types.

    A `BaseFieldGroup` manages a collection of `Field` objects, all assumed to
    be of a consistent type defined by the concrete subclass (e.g.,
    `AngleFieldGroup` manages fields representing angles). It provides an
    interface for adding, retrieving, and managing these fields, as well as
    methods for converting field data into ray parameters for optical
    simulations.

    Attributes:
        fields (list[Field]): A list of `Field` objects in the group. Each field
            typically stores x and y coordinates and vignetting factors.
        telecentric (bool): Indicates whether the optical system is telecentric
            in object space. This property can affect how fields are
            interpreted or validated by subclasses. Defaults to `False`.
    """

    def __init__(self) -> None:
        """Initializes a new `BaseFieldGroup` instance."""
        self.fields: list[Field] = []
        self.telecentric: bool = False

    @property
    def x_fields(self) -> be.ndarray:
        """be.ndarray: An array of x-coordinates for all fields in the group.

        Returns `be.empty(0)` if no fields are present.
        """
        if not self.fields:
            return be.empty(0, dtype=be.float64)
        return be.array([field.x for field in self.fields])

    @property
    def y_fields(self) -> be.ndarray:
        """be.ndarray: An array of y-coordinates for all fields in the group.

        Returns `be.empty(0)` if no fields are present.
        """
        if not self.fields:
            return be.empty(0, dtype=be.float64)
        return be.array([field.y for field in self.fields])

    @property
    def max_x_field(self) -> float:
        """float: Maximum absolute field value in the x-direction.

        Returns 0.0 if no fields are present.
        """
        if not self.fields:
            return 0.0
        return float(be.max(be.abs(self.x_fields)))

    @property
    def max_y_field(self) -> float:
        """float: Maximum absolute field value in the y-direction.

        Returns 0.0 if no fields are present.
        """
        if not self.fields:
            return 0.0
        return float(be.max(be.abs(self.y_fields)))

    @property
    def max_field(self) -> float:
        """float: Maximum radial field value (sqrt(x^2 + y^2)).

        Returns 0.0 if no fields are present.
        """
        if not self.fields:
            return 0.0
        if (
            self.x_fields.size == 0 and self.y_fields.size == 0
        ):  # Should not happen if fields is not empty
            return 0.0
        return float(be.max(be.sqrt(self.x_fields**2 + self.y_fields**2)))

    @property
    def num_fields(self) -> int:
        """int: The number of fields currently in the group."""
        return len(self.fields)

    @property
    def vx(self) -> be.ndarray:
        """be.ndarray: An array of x-vignetting factors for each field.

        Returns `be.empty(0)` if no fields are present.
        """
        if not self.fields:
            return be.empty(0, dtype=be.float64)
        return be.array([field.vx for field in self.fields])

    @property
    def vy(self) -> be.ndarray:
        """be.ndarray: An array of y-vignetting factors for each field.

        Returns `be.empty(0)` if no fields are present.
        """
        if not self.fields:
            return be.empty(0, dtype=be.float64)
        return be.array([field.vy for field in self.fields])

    @abstractmethod
    def validate(self, optic: "Optic") -> None:
        """Validates the field group configuration against the optic.

        Subclasses should implement this method to perform specific validation,
        such as checking compatibility with the optic's telecentricity or
        object distance.

        Args:
            optic (Optic): The optical system instance to validate against.

        Raises:
            ValueError: If the validation fails.
        """
        pass  # pragma: no cover

    @abstractmethod
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
        """Converts norm. field and pupil coords to ray origins and directions.

        This method is crucial for generating rays for tracing. Subclasses must
        implement the specific logic for their field type (e.g., angle, height)
        to correctly determine the starting position (ro) and direction cosines
        (rd) of rays originating from the object surface or its equivalent.

        Args:
            Hx (be.ndarray): Normalized x-coordinates of the field points.
            Hy (be.ndarray): Normalized y-coordinates of the field points.
            Px (be.ndarray): Normalized x-coordinates of the pupil points.
            Py (be.ndarray): Normalized y-coordinates of the pupil points.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.
            optic (Optic): The optical system, providing context like object
                distance and surface information.

        Returns:
            tuple[be.ndarray, be.ndarray]: A tuple containing:
                - ro (be.ndarray): Ray origin coordinates (x, y, z).
                - rd (be.ndarray): Ray direction cosines (l, m, n).
        """
        pass  # pragma: no cover

    @abstractmethod
    def to_paraxial_starting_ray(
        self, Hy: float, Py: float, wavelength: float, optic: "Optic"
    ) -> tuple[float, float]:
        """Calculates starting height and angle for a paraxial ray.

        This method is used for paraxial ray tracing. Subclasses must implement
        the logic to convert their specific field type representation (e.g.,
        angle, height) into an initial height (y) and angle (u) for a paraxial
        ray at the object surface or its equivalent.

        Args:
            Hy (float): Normalized y-coordinate of the field point.
            Py (float): Normalized y-coordinate of the pupil point.
            wavelength (float): The wavelength of the ray in microns.
            optic (Optic): The optical system, providing context like object
                distance and paraxial properties.

        Returns:
            tuple[float, float]: A tuple containing:
                - y (float): Initial height of the paraxial ray.
                - u (float): Initial angle (slope) of the paraxial ray.
        """
        pass  # pragma: no cover

    def get_vig_factor(self, Hx: float, Hy: float) -> tuple[float, float]:
        """Calculates interpolated vignetting factors for a given field position.

        Uses nearest-neighbor interpolation based on the vignetting factors
        defined for the fields in this group. The field coordinates (Hx, Hy)
        are normalized by `self.max_field` before interpolation if `max_field`
        is non-zero.

        Args:
            Hx (float): The normalized x-component of the field for which to
                interpolate vignetting.
            Hy (float): The normalized y-component of the field for which to
                interpolate vignetting.

        Returns:
            tuple[float, float]: A tuple `(vx_new, vy_new)` representing the
            interpolated x and y vignetting factors. If no fields are present,
            returns `(0.0, 0.0)`.
        """
        if not self.fields:
            return 0.0, 0.0

        max_f = self.max_field
        if max_f == 0:
            # If max_field is 0, all fields are at (0,0).
            # Use the vignetting of the first field (or average if desired).
            # Assuming if max_field is 0, there's likely one field at (0,0).
            return float(self.fields[0].vx), float(self.fields[0].vy)

        # Normalize defined field points for interpolation
        norm_x_fields = self.x_fields / max_f
        norm_y_fields = self.y_fields / max_f

        defined_fields = be.stack((norm_x_fields, norm_y_fields), axis=-1)
        vignetting_data = be.stack((self.vx, self.vy), axis=-1)

        # Ensure Hx and Hy are within the range of defined normalized fields
        # or handle extrapolation if be.nearest_nd_interpolator requires it.
        # For nearest neighbor, it effectively clips to the nearest point.
        interpolated_v = be.nearest_nd_interpolator(
            defined_fields, vignetting_data, be.array([Hx]), be.array([Hy])
        )
        vx_new = float(interpolated_v[0, 0])
        vy_new = float(interpolated_v[0, 1])
        return vx_new, vy_new

    def get_field_coords(self) -> list[tuple[float, float]]:
        """Returns normalized coordinates of the fields.

        If `self.max_field` is 0 (e.g., only one field at the origin or no
        fields), it returns `[(0.0, 0.0)]` if fields exist, or `[]` if no
        fields. Otherwise, it calculates the normalized coordinates
        (x/max_field, y/max_field) for each field.

        Returns:
            list[tuple[float, float]]: A list of tuples, where each tuple
            contains the `(normalized_x, normalized_y)` coordinates of a field.
            Returns an empty list if no fields are present.
        """
        if not self.fields:
            return []

        max_f = self.max_field
        if max_f == 0:
            # All fields are at (0,0) or only one field at (0,0)
            return [(0.0, 0.0)] * len(self.fields)

        return [
            (float(field.x / max_f), float(field.y / max_f)) for field in self.fields
        ]

    def add_field(self, field: Field, optic: "Optic") -> None:
        """Adds a `Field` instance to the group after validation.

        The provided `optic` instance is used by the `validate` method to
        ensure the field group's configuration is compatible with the overall
        optical system settings before adding the field.

        Args:
            field (Field): The `Field` object to be added.
            optic (Optic): The `Optic` instance, used for validation purposes.
                           The `validate` method might check for consistencies
                           like field type compatibility with telecentric
                           settings.

        Raises:
            ValueError: If `self.validate(optic)` raises an error.
        """
        self.validate(optic)  # Validate before adding
        self.fields.append(field)

    def get_field(self, field_number: int) -> Field:
        """Retrieves the field at the specified index.

        Args:
            field_number (int): The index of the field to retrieve.

        Returns:
            Field: The `Field` object at the specified index.

        Raises:
            IndexError: If `field_number` is out of range.
        """
        if not 0 <= field_number < len(self.fields):
            raise IndexError(
                f"Field number {field_number} is out of range. "
                f"Number of fields: {len(self.fields)}"
            )
        return self.fields[field_number]

    def set_telecentric(self, is_telecentric: bool) -> None:
        """Sets the object space telecentricity flag for the field group.

        Subclasses might override this or use this flag in their `validate`
        method to ensure compatibility (e.g., an `AngleFieldGroup` might
        disallow telecentricity).

        Args:
            is_telecentric (bool): `True` if the system is telecentric in object
                space, `False` otherwise.
        """
        self.telecentric = is_telecentric

    @classmethod
    @abstractmethod
    def field_type_string(cls) -> str:
        """Returns the string identifier for this field group type.

        This is used for serialization and by the factory function.
        Example: "angle", "object_height".

        Returns:
            str: The type string.
        """
        pass  # pragma: no cover

    def to_dict(self) -> dict[str, Any]:
        """Serializes the field group to a dictionary.

        The dictionary includes the list of serialized fields, the telecentric
        status, and the field type string.

        Returns:
            dict[str, Any]: A dictionary representation of the field group,
            suitable for JSON serialization.
        """
        return {
            "field_type": self.field_type_string(),
            "fields": [field.to_dict() for field in self.fields],
            "telecentric": self.telecentric,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], optic_for_add_field: "Optic"
    ) -> "BaseFieldGroup":
        """Creates a `BaseFieldGroup` (or subclass) instance from a dictionary.

        This method is intended to be called on the specific subclass
        (e.g., `AngleFieldGroup.from_dict(...)`). It reconstructs the field
        group, adds fields from the dictionary, and sets the telecentric flag.

        Note:
            The `optic_for_add_field` argument is required because `add_field`
            performs validation against the optic. This introduces a dependency
            that might be refactored in the future if validation logic changes.
            If `optic_for_add_field` is `None`, field addition might skip
            validation or use a default validation context, which could be
            risky. For robust reconstruction, a valid `Optic` context is
            preferred.

        Args:
            data (dict[str, Any]): A dictionary representation of the field
                group, typically obtained from `to_dict()`. Expected keys
                include 'fields' (a list of field dictionaries) and
                'telecentric' (a boolean). The 'field_type' key from `data` is
                used by a factory, not directly here.
            optic_for_add_field (Optic): The Optic instance to be used for
                validation when adding fields. This is a bit problematic as
                `from_dict` is a class method. Ideally, validation happens
                after full construction.

        Returns:
            BaseFieldGroup: An instance of the calling class
            (e.g., `AngleFieldGroup`) populated with data from the dictionary.

        Raises:
            ValueError: If required keys are missing in `data` or if
                `Field.from_dict` fails.
        """
        # This method will be implemented by concrete subclasses, but they can
        # call super().from_dict(...) or replicate this logic.
        # The `cls()` call instantiates the specific subclass.
        field_group = cls()
        field_dicts = data.get("fields")
        if field_dicts is None:
            raise ValueError("Field data is missing in 'fields' key.")

        for field_dict in field_dicts:
            # We are removing 'field_type' from Field instances.
            # Field.from_dict should be updated to not expect it.
            # The validation in add_field needs the optic.
            # This is a tricky part of from_dict when methods need external
            # context. For now, we pass it. If optic_for_add_field is None,
            # validate might need to handle it.
            field_group.add_field(Field.from_dict(field_dict), optic_for_add_field)

        telecentric_val = data.get("telecentric")
        if telecentric_val is None:
            raise ValueError("Telecentric flag is missing in 'telecentric' key.")
        field_group.set_telecentric(telecentric_val)
        return field_group
