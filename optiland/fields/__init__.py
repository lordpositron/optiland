"""Fields Subpackage.

This package provides classes for managing field points in optical systems.

It includes:
    - `Field`: Represents a single field point with coordinates and vignetting.
    - `BaseFieldGroup`: An abstract base class for collections of fields of the
      same type.
    - `AngleFieldGroup`: A concrete implementation for fields defined by angles.
    - `make_field_group`: A factory function to create specific `FieldGroup`
      instances based on a type string.

The `make_field_group` factory is the preferred way to create `FieldGroup`
instances within the `Optic` class, as it decouples `Optic` from concrete
field group implementations. To add a new field group type (e.g., `ObjectHeightFieldGroup`),
it needs to be:
1. Implemented as a subclass of `BaseFieldGroup`.
2. Added to the `_FIELD_GROUP_CONSTRUCTORS` dictionary within this `__init__.py` file
   so `make_field_group` can find and instantiate it.
"""

from typing import Dict, Type

from .angle_group import AngleFieldGroup
from .base_group import BaseFieldGroup
from .field import Field

# Dictionary to map field type strings to their constructor classes.
# This acts as a registry for available FieldGroup types.
_FIELD_GROUP_CONSTRUCTORS: Dict[str, Type[BaseFieldGroup]] = {
    AngleFieldGroup.field_type_string(): AngleFieldGroup,
    # To add new field group types, add them here, e.g.:
    # ObjectHeightFieldGroup.field_type_string(): ObjectHeightFieldGroup,
}


def make_field_group(field_type_str: str) -> BaseFieldGroup:
    """Factory function to create a `FieldGroup` instance of a specific type.

    This function looks up the `field_type_str` in a registry of known
    field group types and returns an instance of the corresponding class.

    Args:
        field_type_str (str): The string identifier for the desired field group
            type (e.g., "angle", "object_height"). These strings are typically
            defined by the `field_type_string()` class method of each
            `BaseFieldGroup` subclass.

    Returns:
        BaseFieldGroup: An instance of the appropriate `BaseFieldGroup` subclass
        (e.g., `AngleFieldGroup`).

    Raises:
        ValueError: If the `field_type_str` is not recognized (i.e., not found
            in the registry).
    """
    constructor = _FIELD_GROUP_CONSTRUCTORS.get(field_type_str)
    if constructor:
        return constructor()
    else:
        valid_types = list(_FIELD_GROUP_CONSTRUCTORS.keys())
        raise ValueError(
            f"Unknown field type: '{field_type_str}'. "
            f"Valid types are: {valid_types}"
        )


__all__ = [
    "Field",
    "BaseFieldGroup",
    "AngleFieldGroup",
    "make_field_group",
]
