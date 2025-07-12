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


class ObjectHeightMode(BaseFieldMode):
    """Field mode for fields defined by object height.

    This mode implements field-dependent calculations assuming the field
    points are specified as absolute heights on the object surface.

    It is typically used for finite object conjugates where the object size is
    given directly.
    """

    def __init__(self):
        """Initializes an ObjectHeightMode."""
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
        if optic.obj_space_telecentric:
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


class AngleMode(BaseFieldMode):
    """Field mode for fields defined by an angle.

    This mode implements field-dependent calculations assuming the field
    points are specified as angles relative to the optical axis, typically
    from the perspective of the entrance pupil.

    It is commonly used for objects at infinity or when field coverage is
    naturally expressed in angular terms.
    """

    def __init__(self):
        """Initializes an AngleMode."""
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
            if optic.obj_space_telecentric:
                # Also caught by validate_optic_state
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity."
                )
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()

            z_surf_internal = optic.surface_group.positions[1:-1]
            offset_val = optic.paraxial.EPD()
            starting_z_offset = offset_val - (
                be.min(z_surf_internal) if z_surf_internal.size > 0 else 0
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
            x0_scalar = -be.tan(be.radians(field_x_angle_deg)) * (EPL - z_obj_global)
            y0_scalar = -be.tan(be.radians(field_y_angle_deg)) * (EPL - z_obj_global)
            z0_scalar = z_obj_global  # Rays start on the object surface.

            x0 = be.full_like(Px, x0_scalar)
            y0 = be.full_like(Px, y0_scalar)
            z0 = be.full_like(Px, z0_scalar)

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
        if optic.obj_space_telecentric:
            raise ValueError(
                'Field type "angle" is invalid for telecentric object space.'
            )
        pass


class ImageSpaceMode(BaseFieldMode):
    """Field mode for fields defined by image height.

    This mode acts as a wrapper, utilizing a specified field solver
    (paraxial or real) and an underlying object-space field mode
    (e.g., AngleMode, ObjectHeightMode). It calculates the object-space
    field parameter (e.g., angle or height) that corresponds to a desired
    image height, and then delegates the ray generation to the underlying
    object-space mode.

    Attributes:
        solver (BaseFieldSolver): The solver instance used to find the
            object-space field equivalent for a given image height.
        base_mode (BaseFieldMode): The underlying object-space
            mode used for actual ray calculations.
    """

    def __init__(self, solver, base_mode):
        """Initializes an ImageSpaceMode.

        Args:
            solver (BaseFieldSolver): An instance of a field solver (e.g.,
                ParaxialFieldSolver, RealFieldSolver) used to determine the
                object-space field value that produces the desired image height.
            base_mode (BaseFieldMode): An instance of an object-space
                field mode (e.g., AngleMode, ObjectHeightMode) that will
                be used to generate rays once the equivalent object-space field
                is known.
        """
        self.solver = solver
        self.base_mode = base_mode

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate ray origin coordinates for image height fields.

        This method first calculates the target image height. Then, it uses the
        provided solver to find the equivalent object-space field (e.g., angle
        or height) that would produce this target image height. Finally, it
        delegates the ray origin calculation to the underlying base_mode,
        temporarily setting the optic's max_field to this equivalent
        object-space field value and using a normalized field input of 1.0.

        Args:
            optic (Optic): The optical system instance.
            Hx (float): Normalized x-coordinate of the field point. This is
                ignored as image height is typically 1D (y-dimension).
                The solver will operate on Hy.
            Hy (float): Normalized y-coordinate of the field point, used to
                determine the target image height.
            Px (float or be.ndarray): Normalized x-coordinate(s) on the pupil.
            Py (float or be.ndarray): Normalized y-coordinate(s) on the pupil.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: A tuple (x0, y0, z0)
            representing the ray origin coordinates.

        Raises:
            NotImplementedError: If Hx is non-zero, as image space fields are
                currently assumed to be defined only in the y-dimension.
        """
        if not be.isclose(Hx, 0.0):
            raise NotImplementedError(
                "ImageSpaceMode currently only supports Hy (1D image height)."
            )

        target_image_height = optic.fields.max_field * Hy

        # Use the solver to find the object-space field value (e.g., angle or height)
        # that produces the target_image_height.
        # The solver typically needs the optic and the target image height.
        equivalent_object_field = self.solver.solve(optic, target_image_height)

        # Store original max_field and temporarily set it for the base mode call.
        original_max_field = optic.fields.max_field
        # original_max_y_field was here, removed as unused in this method.

        try:
            # The task specifies using the equivalent_object_field as the temporary
            # max_field for the base_mode, and calling the base_mode
            # with a normalized field coordinate of 1.0.

            # Interpretation:
            # 1. `solver.solve(optic, target_image_height)` returns the actual
            #    object-space field value (e.g., -5 degrees or 10mm). This value
            #    can be positive or negative.
            # 2. `optic.fields.max_field` is temporarily set to this exact
            #    `equivalent_object_field` value.
            # 3. `base_mode.get_ray_origins` is called with `Hy = 1.0`.
            #    The base mode will then calculate `max_field * Hy`, which
            #    effectively becomes `equivalent_object_field * 1.0`, yielding
            #    the desired object-space field parameter for ray generation.
            #    This works if base modes correctly use `max_field * Hy`
            #    (e.g., AngleMode: `max_field_angle_deg = max_field * Hy`;
            #    ObjectHeightMode: `field_y = max_field * Hy`).
            #    This seems to be the case from reviewing their implementations.

            optic.fields.max_field = equivalent_object_field

            # For 2D fields (if supported in future), Hx would also be handled.
            # Currently, Hx is 0 for image space fields as per NotImplementedError.
            base_Hx = 0.0
            # As per instruction "using a normalized field coordinate of 1.0"
            base_Hy = 1.0

            return self.base_mode.get_ray_origins(
                optic, base_Hx, base_Hy, Px, Py, vx, vy
            )
        finally:
            # Restore original max_field.
            optic.fields.max_field = original_max_field

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculate paraxial object position for image height fields.

        This method follows a similar pattern to get_ray_origins:
        1. Calculate target image height.
        2. Use the solver to find the equivalent object-space field value.
        3. Delegate to the base_mode's get_paraxial_object_position,
           temporarily setting optic's max_field to the equivalent value and
           using a normalized field input of 1.0.

        Args:
            optic (Optic): The optical system instance.
            Hy (float): Normalized y-coordinate of the field point.
            y1 (be.ndarray): Ray height(s) at the entrance pupil.
            EPL (float): Entrance Pupil Location (axial position).

        Returns:
            tuple[be.ndarray, be.ndarray]: Paraxial object coordinates (y0, z0).
        """
        target_image_height = optic.fields.max_field * Hy
        equivalent_object_field = self.solver.solve(optic, target_image_height)

        original_max_field = optic.fields.max_field
        try:
            optic.fields.max_field = equivalent_object_field
            # Base mode called with Hy_norm=1.0
            base_Hy = 1.0
            return self.base_mode.get_paraxial_object_position(optic, base_Hy, y1, EPL)
        finally:
            optic.fields.max_field = original_max_field

    def get_chief_ray_start_params(
        self, optic, chief_ray_y_at_stop, chief_ray_u_at_stop
    ):
        """Calculate chief ray start parameters for image height fields.

        This method also delegates to the base_mode. However, the concept
        of "max_field" for chief ray calculation needs careful handling.
        The base mode's `get_chief_ray_start_params` typically uses
        `optic.fields.max_y_field`. We need to ensure this reflects the
        object-space equivalent of the *maximum defined image height* for
        the optic.

        The `equivalent_object_field` determined by the solver should be
        for the *actual* `optic.fields.max_field` (which defines max image height).

        Args:
            optic (Optic): The optical system instance.
            chief_ray_y_at_stop (float): Paraxial ray height at the initial plane.
            chief_ray_u_at_stop (float): Paraxial ray slope at the initial plane.

        Returns:
            float: The adjusted starting slope `u1` for the chief ray trace.
        """
        # For ImageSpaceMode, `optic.fields.max_field` (or `max_y_field` if distinct)
        # defines the maximum *image height* for the system.
        # The base mode's `get_chief_ray_start_params` uses
        # `optic.fields.max_y_field` to refer to the maximum *object-space*
        # field (e.g., max angle or max object height).
        #
        # Therefore, we need to:
        # 1. Get the maximum defined image height for the optic.
        #    Let's assume `optic.fields.max_y_field` is used for this,
        #    as it's the one base modes query for their max object field.
        #    If `max_y_field` is not specifically set for image height, it
        #    would default to `max_field`.
        # 2. Use the solver to find the object-space field value that
        #    corresponds to this maximum image height.
        # 3. Temporarily set `optic.fields.max_y_field` to this *absolute*
        #    equivalent object-space field value before calling the
        #    base_mode's method. Base modes expect a positive extent.

        # Use max_y_field as it's what base modes (Angle, ObjectHeight) query.
        # This `max_y_field` for an ImageSpaceMode optic defines the max image height.
        max_defined_image_height = optic.fields.max_y_field

        # Find object-space field equivalent to this max image height.
        # The solver should return the actual field value (can be signed).
        equivalent_object_field_for_max_image = self.solver.solve(
            optic, max_defined_image_height
        )

        # Store original values to restore them.
        original_max_y_field = optic.fields.max_y_field
        # Note: `optic.fields.max_field` is not directly used by base mode's
        # `get_chief_ray_start_params`, so we only need to manage `max_y_field`.

        try:
            # Base modes expect `optic.fields.max_y_field` to be a positive
            # value representing the maximum extent of the object-space field.
            optic.fields.max_y_field = be.fabs(equivalent_object_field_for_max_image)

            return self.base_mode.get_chief_ray_start_params(
                optic, chief_ray_y_at_stop, chief_ray_u_at_stop
            )
        finally:
            # Restore original max_y_field.
            optic.fields.max_y_field = original_max_y_field
            # optic.fields.max_field remains untouched if it wasn't modified.

    def validate_optic_state(self, optic):
        """Validate if the optic's state is compatible with ImageSpaceMode.

        This primarily delegates validation to the underlying base_mode.
        The ImageSpaceMode wrapper itself imposes few additional constraints
        beyond those of its components (solver and base_mode).

        The solver might have its own validation (e.g., requiring computable
        paraxial properties).

        Args:
            optic (Optic): The optical system instance to validate.

        Raises:
            ValueError: If the optic's configuration is incompatible.
        """
        # First, allow the base mode to validate itself.
        if self.base_mode:
            self.base_mode.validate_optic_state(optic)

        # Solver validation:
        # Solvers might have implicit requirements (e.g., ParaxialFieldSolver
        # needs a valid paraxial system). These are typically checked when
        # solver methods are called or paraxial properties are accessed.
        # No explicit, additional validation call to the solver is made here,
        # assuming failures will be caught during `solver.solve()`.
        pass

    def __repr__(self):
        """Return a string representation of the ImageSpaceMode."""
        return (
            f"{self.__class__.__name__}("
            f"solver={self.solver!r}, base_mode={self.base_mode!r})"
        )
