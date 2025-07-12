"""Field Solvers Module

This module contains classes for solving object-space field values
(e.g., height or angle) corresponding to a desired image-space height.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod

import numpy as np

from .strategies import ImageSpaceField


class BaseFieldSolver(ABC):
    """
    Abstract base class for field solvers.

    A field solver calculates the required object-space field value
    (e.g., height or angle) that corresponds to a desired image-space height.
    """

    @abstractmethod
    def solve(self, optic, target_image_height: float) -> float:
        """
        Calculates the object-space field value for a target image height.

        Args:
            optic: The optical system.
            target_image_height: The desired height on the image surface.

        Returns:
            The corresponding object-space field value.
        """
        pass


class ParaxialFieldSolver(BaseFieldSolver):
    """
    Solves for the object-space field using paraxial ray tracing.
    """

    def solve(self, optic, target_image_height: float) -> float:
        """
        Calculates the object-space field value for a target image height
        using a paraxial ray tracing approach that avoids recursion with
        chief_ray calculations.

        Args:
            optic: The optical system.
            target_image_height: The desired height on the image surface.

        Returns:
            The corresponding object-space field value.
        """
        if not optic.fields:
            raise ValueError("Optic has no fields defined.")

        # Step 1: Determine Base Field Strategy
        field_type_strategy = optic.field_type
        if isinstance(field_type_strategy, ImageSpaceField):
            base_object_field_strategy = field_type_strategy.base_strategy
        else:
            base_object_field_strategy = field_type_strategy

        if not base_object_field_strategy or not hasattr(
            base_object_field_strategy, "get_chief_ray_start_params"
        ):
            raise ValueError(
                "Could not determine a valid base object field strategy "
                "with get_chief_ray_start_params method."
            )

        # Step 2: Simulate Image Height for a Unit Object Field
        unit_object_field_value = 1.0  # e.g., 1mm or 1 degree
        wavelength = optic.primary_wavelength
        stop_index = optic.surface_group.stop_index

        # 2a: Initial backward trace from stop (arbitrary slope)
        y_at_stop_center_for_bwd_arb = 0.0
        u_backward_arbitrary = 0.1  # Standard arbitrary slope
        # z0 for reverse trace is distance from last surface to stop surface
        z_coords_for_reverse_trace = (
            optic.surface_group.positions[-1]
            - optic.surface_group.positions[stop_index]
        )
        num_surfaces_to_skip_rev = optic.surface_group.num_surfaces - stop_index

        # Use optic.paraxial._trace_generic for tracing
        y_obj_parts_arb, u_obj_parts_arb = optic.paraxial._trace_generic(
            y_at_stop_center_for_bwd_arb,
            u_backward_arbitrary,
            z_coords_for_reverse_trace,
            wavelength,
            reverse=True,
            skip=num_surfaces_to_skip_rev,
        )
        y_obj_at_ref_backward = y_obj_parts_arb[-1]
        u_obj_at_ref_backward = u_obj_parts_arb[-1]

        # 2b: Get chief ray start params for the unit object field
        original_max_y_field = optic.fields.max_y_field
        try:
            # Temporarily set max_y_field to our unit value for the call
            optic.fields.max_y_field = unit_object_field_value
            # The strategy's get_chief_ray_start_params will use this temp value
            u_start_for_unit_field_obj_space = (
                base_object_field_strategy.get_chief_ray_start_params(
                    optic, y_obj_at_ref_backward, u_obj_at_ref_backward
                )
            )
        finally:
            optic.fields.max_y_field = original_max_y_field

        # 2c: Determine actual y_start and u_start for the forward trace
        # This step mimics the second reverse trace in paraxial.chief_ray
        # to get the y,u at object plane for the calculated
        # u_start_for_unit_field_obj_space

        # `u_start_for_unit_field_obj_space` is the slope that should emerge from the
        # object side if the ray were traced backward from the stop, corresponding
        # to the unit field.
        # Now, we need the y,u at the object plane for a *forward* trace.
        # paraxial.chief_ray does:
        #   yn_rev, un_rev = self._trace_generic(y0_val, u1_chief_start, z0_rev_trace,
        # reverse=True, skip=skip)
        #   y_fwd = -yn_rev[-1,0]
        #   u_fwd = un_rev[-1,0]
        # So, u1_chief_start is `u_start_for_unit_field_obj_space`

        y_obj_final_parts, u_obj_final_parts = optic.paraxial._trace_generic(
            y_at_stop_center_for_bwd_arb,  # y is 0 at stop center
            u_start_for_unit_field_obj_space,  # This is u at the stop for reverse trace
            z_coords_for_reverse_trace,
            wavelength,
            reverse=True,
            skip=num_surfaces_to_skip_rev,
        )

        actual_y_start_for_unit_field = -y_obj_final_parts[-1]
        actual_u_start_for_unit_field = u_obj_final_parts[-1]

        # Define z_start for forward trace (position of the first optical surface,
        # or object plane if finite and explicitly defined before surface 1)
        # paraxial.chief_ray uses z0_fwd_trace = self.optic.surface_group.positions[1]
        # This assumes the forward trace starts *at* the first surface.
        # If object is at finite distance, ray starts at object plane z.
        # Let's check object_surface.is_infinite
        if optic.object_surface.is_infinite:
            # For infinite object, paraxial rays often start at/just before the first
            # surface.
            # The actual_y_start and actual_u_start are usually defined at this plane.
            z_start_fwd_trace = optic.surface_group.positions[1]
        else:
            # For finite object, ray starts at the object surface.
            z_start_fwd_trace = optic.object_surface.geometry.cs.z
            # We need to propagate actual_y_start_for_unit_field and
            # actual_u_start_for_unit_field
            # from surface 1 (where they are currently defined if we followed
            # chief_ray's reverse trace logic)
            # to z_start_fwd_trace if z_start_fwd_trace is not positions[1].
            # The values from reverse trace ending at object space are already at the
            # object "plane"
            # (or reference plane for infinite).
            # The `paraxial.chief_ray` forward trace
            # `_trace_generic(y,u,z_start_fwd_trace)`
            # implies y,u are known *at* z_start_fwd_trace.
            # The yn_rev[-1], un_rev[-1] are values at the object side reference (often
            #  surf 1 for infinite, obj plane for finite).
            # So, if z_start_fwd_trace is object_surface.z, and yn_rev[-1] was defined
            # there, it's fine.
            # The _trace_generic with reverse=True, when traced up to object space
            # (skip= appropriate number),
            # should yield y, u at the object plane or equivalent first reference
            #  surface.
            # Assuming actual_y/u_start are correctly defined at the object reference
            # plane for forward trace.
            pass

        # 2d: Perform Forward Trace for Unit Object Field
        y_trace_unit, _ = optic.paraxial._trace_generic(
            actual_y_start_for_unit_field,
            actual_u_start_for_unit_field,
            z_start_fwd_trace,  # Starting z position for the forward trace
            wavelength,
            reverse=False,
            skip=0,  # Trace through all surfaces from start
        )
        y_image_for_unit_object_field = y_trace_unit[-1]

        # Step 3: Calculate Scale Factor and Solved Object Field
        if np.isclose(y_image_for_unit_object_field, 0.0):
            if np.isclose(target_image_height, 0.0):
                return 0.0
            else:
                raise ValueError(
                    "Unit object field results in zero image height, "
                    "cannot scale to non-zero target."
                )

        scale_factor = target_image_height / y_image_for_unit_object_field
        solved_object_field = unit_object_field_value * scale_factor

        return solved_object_field
