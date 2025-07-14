"""Field Strategies Module

This module defines concrete field modes classes for Optiland, implementing
a strategy pattern for handling different field type behaviors (e.g., object
height vs. angle). These modes encapsulate the logic for calculations
such as ray origin determination and paraxial object positioning based on the
chosen field definition.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod

import optiland.backend as be


class BaseFieldMode(ABC):
    """
    Abstract base class for defining field modes.
    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseFieldMode._registry[cls.__name__] = cls

    @abstractmethod
    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """
        Calculates the origin points for rays based on field type.
        """
        pass

    @abstractmethod
    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """
        Calculates the paraxial object position based on field type.
        """
        pass

    @abstractmethod
    def get_chief_ray_start_params(
        self, optic, chief_ray_y_at_stop, chief_ray_u_at_stop
    ):
        """
        Calculates the starting parameters for the chief ray based on field type.
        """
        pass

    @abstractmethod
    def validate_optic_state(self, optic):
        """
        Validates the optic state based on the field type.
        """
        pass

    def to_dict(self):
        """Convert the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data):
        """Create a field mode from a dictionary.

        Args:
            data (dict): A dictionary containing the field mode data, including
                its 'type'.

        Returns:
            BaseFieldMode: An instance of a specific geometry subclass created
            from the dictionary data.

        """
        field_mode_type = data.get("type")
        if field_mode_type not in cls._registry:
            raise ValueError(f"Unknown field mode type: {field_mode_type}")

        # Delegate to the correct subclass's from_dict
        return cls._registry[field_mode_type]()


class ObjectHeightFieldMode(BaseFieldMode):
    """Field mode for fields defined by object height.

    This mode implements field-dependent calculations assuming the field
    points are specified as absolute heights on the object surface.

    It is typically used for finite object conjugates where the object size is
    given directly.
    """

    def __init__(self):
        """Initializes an ObjectHeightMode."""
        self.type_ = "object_height"
        super().__init__()

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate ray origin coordinates for object height fields.

        For finite objects, the origin is determined by the field height
        (Hx, Hy scaled by max_field) on the object surface, including its sag.

        Args:
            optic (Optic): The optical system instance.
            Hx (float): Normalized x-coordinate of the field point.
            Hy (float): Normalized y-coordinate of the field point.
            Px (float or be.ndarray): Normalized x-coordinate(s) on the pupil.
            Py (float or be.ndarray): Normalized y-coordinate(s) on the pupil.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: A tuple (x0, y0, z0)
            representing the ray origin coordinates. Each element is an array
            broadcastable to the shape of Px/Py.

        Raises:
            ValueError: If this mode is used with an object at infinity,
                as object height is not a valid definition in that case.

        """
        obj = optic.object_surface
        max_field = optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy

        if obj.is_infinite:
            # This check is also in validate_optic_state but provides a safeguard.
            raise ValueError(
                'Field type "object_height" cannot be used for an object at infinity.'
            )
        else:
            x0 = be.array(field_x)
            y0 = be.array(field_y)
            # Calculate sag at the object surface for the given field point
            z0_sag = obj.geometry.sag(x0, y0)
            z0 = z0_sag + obj.geometry.cs.z  # Add global z-position of object surface

            # Ensure outputs are broadcastable with pupil coordinates
            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculate paraxial object position for object height fields.

        The paraxial object position (y0, z0) is determined based on the
        normalized field height Hy. For object height definition, y0 is the
        actual field height and z0 is the object surface's axial position.

        Args:
            optic (Optic): The optical system instance.
            Hy (float): Normalized y-coordinate of the field point.
            y1 (be.ndarray): Ray height(s) at the entrance pupil.
            EPL (float): Entrance Pupil Location (axial position).

        Returns:
            tuple[be.ndarray, be.ndarray]: Paraxial object coordinates (y0, z0).
            Each element is an array broadcastable to the shape of y1.

        Raises:
            ValueError: If this mode is used with an object at infinity.

        """
        obj = optic.object_surface
        field_y = optic.fields.max_field * Hy

        if obj.is_infinite:
            raise ValueError(
                'Field type "object_height" cannot be used for an object at infinity.'
            )
        else:
            y0_scalar = -field_y  # Object height (y-coordinate)
            z0_scalar = obj.geometry.cs.z  # Object's axial position

            y0 = be.ones_like(y1) * y0_scalar
            z0 = be.ones_like(y1) * z0_scalar
        return y0, z0

    def get_chief_ray_start_params(
        self, optic, chief_ray_y_at_stop, chief_ray_u_at_stop
    ):
        """Calculate starting slope for chief ray tracing (object height fields).

        This method determines the initial slope `u1` for a paraxial ray trace
        (typically run in reverse from the stop) that will correspond to the
        chief ray for the maximum field defined by object height.

        Args:
            optic (Optic): The optical system instance.
            chief_ray_y_at_stop (float): Paraxial ray height at the initial plane
                (e.g., object plane) after a reverse trace from stop center.
                Corresponds to `y[-1]` from that reverse trace.
            chief_ray_u_at_stop (float): Paraxial ray slope at the initial plane
                after a reverse trace from stop center. Corresponds to `u[-1]`
                from that reverse trace. (Not used by this mode).

        Returns:
            float: The adjusted starting slope `u1` for the chief ray trace.

        """
        max_field = optic.fields.max_y_field  # Maximum y-field height
        # The 0.1 is a scaling factor and is arbitrary
        u1 = 0.1 * max_field / chief_ray_y_at_stop
        return u1

    def validate_optic_state(self, optic):
        """Validate if the optic's state is compatible with ObjectHeightMode.

        Checks include:
        - Object must not be at infinity.
        - If object space is telecentric, aperture type cannot be EPD or imageFNO.

        Args:
            optic (Optic): The optical system instance to validate.

        Raises:
            ValueError: If the optic's configuration is incompatible.

        """
        if optic.object_surface.is_infinite:
            raise ValueError(
                'Field type "object_height" is invalid for an object at infinity.'
            )
        if optic.fields.telecentric:
            if optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type "EPD" is invalid for telecentric object space '
                    'with "object_height" field type.'
                )
            if optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type "imageFNO" is invalid for telecentric object space '
                    'with "object_height" field type.'
                )


class AngleFieldMode(BaseFieldMode):
    """Field mode for fields defined by an angle.

    This mode implements field-dependent calculations assuming the field
    points are specified as angles relative to the optical axis, typically
    from the perspective of the entrance pupil.

    It is commonly used for objects at infinity or when field coverage is
    naturally expressed in angular terms.
    """

    def __init__(self):
        """Initializes an AngleMode."""
        self.type_ = "angle"
        super().__init__()

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate ray origin coordinates for angle fields.

        For infinite objects, origins are calculated on a plane offset from the
        first surface, based on field angles and pupil parameters.
        For finite objects, origins are calculated based on field angles
        relative to the entrance pupil from the object's z-position.

        Args:
            optic (Optic): The optical system instance.
            Hx (float): Normalized x-coordinate of the field point (angle).
            Hy (float): Normalized y-coordinate of the field point (angle).
            Px (float or be.ndarray): Normalized x-coordinate(s) on the pupil.
            Py (float or be.ndarray): Normalized y-coordinate(s) on the pupil.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: A tuple (x0, y0, z0)
            representing the ray origin coordinates. Each element is an array
            broadcastable to the shape of Px/Py.

        Raises:
            ValueError: If object space is telecentric and object is at infinity
                (an incompatible setup also caught by validate_optic_state).

        """
        obj = optic.object_surface
        max_field = optic.fields.max_field  # Max field angle in degrees
        field_x_angle_deg = max_field * Hx
        field_y_angle_deg = max_field * Hy

        if obj.is_infinite:
            if optic.fields.telecentric:
                # Also caught by validate_optic_state
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity."
                )
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()

            z_surf_internal = optic.surface_group.positions[1:-1]
            offset_val = optic.paraxial.EPD()
            starting_z_offset = offset_val - (
                be.min(z_surf_internal) if be.size(z_surf_internal) > 0 else 0
            )

            # Ray starting plane z-coordinate relative to first optical surface
            z_start_plane = optic.surface_group.positions[1] - starting_z_offset

            # Object coordinates (x,y) on this plane to achieve the field angle
            # when viewed from the entrance pupil.
            x_obj_at_plane = -be.tan(be.radians(field_x_angle_deg)) * (
                starting_z_offset + EPL
            )
            y_obj_at_plane = -be.tan(be.radians(field_y_angle_deg)) * (
                starting_z_offset + EPL
            )

            # Ray origins are then pupil points projected onto this plane,
            # offset by the object's position on that plane.
            x0 = Px * EPD / 2 * vx + x_obj_at_plane
            y0 = Py * EPD / 2 * vy + y_obj_at_plane
            z0 = be.full_like(Px, z_start_plane)
        else:  # Finite object
            EPL = optic.paraxial.EPL()
            z_obj_global = obj.geometry.cs.z  # Global z-pos of object surface

            # Calculate object heights (x0, y0) that would produce the given
            # field angles when viewed from the entrance pupil.
            x0 = -be.tan(be.radians(field_x_angle_deg)) * (EPL - z_obj_global)
            y0 = -be.tan(be.radians(field_y_angle_deg)) * (EPL - z_obj_global)
            z0 = z_obj_global  # Rays start on the object surface.

            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)

        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculate paraxial object position for angle fields.

        For infinite objects, y0 is calculated such that a ray starting at
        (y0, z0_first_surface) and passing through (y1, EPL) has the specified
        field angle. z0 is taken as the first optical surface's position.
        For finite objects, y0 is the object height that, from z_obj, makes the
        specified angle towards y1 at the EPL. z0 is the object's axial position.

        Args:
            optic (Optic): The optical system instance.
            Hy (float): Normalized y-coordinate of the field point (angle).
            y1 (be.ndarray): Ray height(s) at the entrance pupil.
            EPL (float): Entrance Pupil Location (axial position).

        Returns:
            tuple[be.ndarray, be.ndarray]: Paraxial object coordinates (y0, z0).
            Each element is an array broadcastable to the shape of y1.

        """
        obj = optic.object_surface
        field_y_angle_deg = optic.fields.max_field * Hy

        if obj.is_infinite:  # Infinite object
            z_first_surf = optic.surface_group.positions[1]
            u0_inf = be.tan(be.radians(field_y_angle_deg))
            y0 = y1 - u0_inf * (EPL - z_first_surf)
            z0 = be.ones_like(y1) * z_first_surf

        else:  # Finite object
            z_obj = obj.geometry.cs.z
            u0_finite = be.tan(be.radians(field_y_angle_deg))
            y0 = y1 - u0_finite * (EPL - z_obj)
            z0 = be.ones_like(y1) * z_obj

        return y0, z0

    def get_chief_ray_start_params(
        self, optic, chief_ray_y_at_stop, chief_ray_u_at_stop
    ):
        """Calculate starting slope for chief ray tracing (angle fields).

        This method determines the initial slope `u1` for a paraxial ray trace
        (typically run in reverse from the stop) that will correspond to the
        chief ray for the maximum field defined by angle.

        Args:
            optic (Optic): The optical system instance.
            chief_ray_y_at_stop (float): Paraxial ray height at the initial plane
                after a reverse trace from stop center. (Not used by this mode).
            chief_ray_u_at_stop (float): Paraxial ray slope at the initial plane
                after a reverse trace from stop center. Corresponds to `u[-1]`
                from that reverse trace.

        Returns:
            float: The adjusted starting slope `u1` for the chief ray trace.

        """
        max_field_angle_deg = optic.fields.max_y_field  # Max y-field angle
        # The 0.1 is a scaling factor and is arbitrary
        u1 = 0.1 * be.tan(be.deg2rad(max_field_angle_deg)) / chief_ray_u_at_stop
        return u1

    def validate_optic_state(self, optic):
        """Validate if the optic's state is compatible with AngleMode.

        Checks include:
        - If object space is telecentric, field type cannot be "angle".

        Args:
            optic (Optic): The optical system instance to validate.

        Raises:
            ValueError: If the optic's configuration is incompatible.

        """
        if optic.fields.telecentric:
            raise ValueError(
                'Field type "angle" is invalid for telecentric object space.'
            )
        pass


class ParaxialImageHeightFieldMode(BaseFieldMode):
    def to_dict(self):
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        data = super().to_dict()
        data["base_mode"] = self.base_mode.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates an ParaxialImageHeightFieldMode instance from a
        dictionary representation.

        Args:
            data (dict): The dictionary representation of the field mode.

        Returns:
            ParaxialImageHeightFieldMode: An instance of ParaxialImageHeightFieldMode.

        """
        if "base_mode" not in data:
            raise ValueError("Missing 'base_mode' key.")

        base_mode = BaseFieldMode.from_dict(data["base_mode"])
        return cls(base_mode)


class RealImageHeightFieldMode(ParaxialImageHeightFieldMode):
    def __init__(self):
        raise NotImplementedError("Real image height field mode not yet implemented.")
