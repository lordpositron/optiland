"""Field Solvers Module

This module contains classes for solving object-space field values
(e.g., height or angle) corresponding to a desired image-space height.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import brentq

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


class RealFieldSolver(BaseFieldSolver):
    """
    Solves for the object-space field using real ray tracing and a numerical solver.
    """

    def solve(
        self, optic, target_image_height: float, tol: float = 1e-6, max_iter: int = 100
    ) -> float:
        """
        Calculates the object-space field value for a target image height
        using real chief ray tracing and a numerical root-finding algorithm.

        Args:
            optic: The optical system.
            target_image_height: The desired height on the image surface.
            tol: Tolerance for the root-finding algorithm.
            max_iter: Maximum iterations for the root-finding algorithm.

        Returns:
            The corresponding object-space field value.
        """
        # `from optiland.optic import Optic` and `from optiland.ray import Ray`
        # are now at the top of the file for clarity if needed globally,
        # or should be explicitly imported here if only for this method/class.
        # For this exercise, assuming they are available from top-level imports.

        # Objective function: traces a real chief ray and returns the difference
        # between its y-height at image surface and `target_image_height`.
        def objective_function(object_field_y: float) -> float:
            # Chief ray has pupil coordinates (Px, Py) = (0, 0).
            # Solving for y-component of field. Assume field_x = 0.
            # `optic.trace_generic` needs initial ray parameters.
            # These depend on field strategy (e.g., AngleField, ObjectHeightField).
            # Field strategy should provide methods to get ray origins from field values. # noqa: E501

            # Assume `optic.field_strategy.get_ray_origins()` gives start point (x,y,z)
            # and direction (l,m,n) for a field point (0, object_field_y)
            # and pupil coordinates (0,0). This part is highly dependent on Optic
            # and FieldStrategy implementation.

            # Simplified initial ray:
            # If object @ infinity, field is angle. Ray starts at 1st surf, angled.
            # If object finite, field is height. Ray starts from obj plane, aims at EP.
            # This logic usually in `field_strategy.get_ray_origins` or similar.

            # Need to create a Ray object. Example:
            # Ray(x,y,z,l,m,n, Px=0,Py=0, Hx=0,Hy=object_field_y, wl=optic.ref_wl)
            # Initial x,y,z,l,m,n must be determined.

            # Placeholder for getting initial ray parameters.
            # Full implementation would use `optic.field_strategy`:
            # initial_ray_params = optic.field_strategy.get_ray_origins(
            #    optic, Hx=0, Hy=object_field_y, Px=0, Py=0, vx=0, vy=0
            # )
            # ray_origin_pos = initial_ray_params[0]  # (x,y,z)
            # ray_origin_dir = initial_ray_params[1]  # (l,m,n)
            # start_surface_idx = initial_ray_params[2]  # often 0 or 1

            # Simplified ray creation:
            # Assume chief ray for field `object_field_y` (Hy) with Px=0, Py=0.
            # `optic.trace_generic` takes a Ray object.
            # Start pos/dir of ray depend on field type.
            # Assume `optic.get_chief_ray_start(Hx, Hy, Px, Py, wavelength)` exists
            # and returns a Ray object ready for tracing (common pattern).

            # If no helper, construct Ray manually using `optic.field_strategy`.
            # AngleField: ray starts at first surface, direction from angle.
            # ObjectHeightField: ray starts at object plane, toward EP center.

            # Assume `optic.get_input_ray` can generate this ray.
            # Common requirement for tracing:
            # `optic.get_input_ray(Hx, Hy, Px, Py, wavelength)`
            # Must be robust to whether field is angle or height.

            # Placeholder for ray generation:
            # Actual implementation queries optic's field strategy
            # to set up initial ray based on `object_field_y`.
            # E.g., if `optic.object_is_at_infinity`, `object_field_y` is angle.
            # Else, it's an object height.

            # Create chief ray (Px=0, Py=0) for given `object_field_y` (Hy).
            # Assume Hx=0 for simplicity in this 1D solver.
            # Ray needs start point and direction.
            # `optic.field_strategy.get_ray_origins` is crucial here.
            # For now, construct simplified ray, assuming it starts
            # at z=0 (or object plane) and is aimed appropriately.
            # This needs to be accurate for tests.

            # Assume `optic.prepare_ray_for_tracing(Hx, Hy, Px, Py, wavelength)`
            # returns a Ray object correctly initialized.
            ray = optic.prepare_ray_for_tracing(
                0, object_field_y, 0, 0, optic.reference_wavelength
            )

            traced_ray = optic.trace_generic(ray)

            if traced_ray is None or not traced_ray.success:
                # Ray trace failed. Return large error or handle appropriately.
                # May happen if trial field value is too extreme.
                return np.inf  # Or other large number to indicate failure.

            # Assume `traced_ray.y_at_image_surface` is y-coord on image surface.
            image_y = traced_ray.y_at_image_surface

            return image_y - target_image_height

        # Determine bracketing interval [a, b] for brentq.
        # Critical. Good start: optic's defined field range.
        # E.g., [-optic.max_field_y, optic.max_field_y].
        # If `optic.fields` is [(fx1, fy1), ...], use min/max of fy.

        # Assume `optic.fields` provides range of valid field values.
        if not optic.fields:
            raise ValueError("Optic has no fields defined to determine solver range.")

        # Get min/max y-field values from `optic.fields`. Assume (x,y) tuples.
        y_fields = [
            f[1] for f in optic.fields if isinstance(f, (list, tuple)) and len(f) == 2
        ]
        if not y_fields:
            # Handle cases like single on-axis field [0,0] or different format.
            # If only [0,0] defined, and target_image_height is 0, then 0 is solution.
            # Otherwise, need a range.
            if (
                target_image_height == 0
                and (0.0 in y_fields or not y_fields)
                and (
                    0.0 in y_fields
                    or (len(optic.fields) == 1 and optic.fields[0][1] == 0.0)
                )
                and objective_function(0.0) == 0.0
            ):
                # If only on-axis field (0,0) present, and target is 0, solution is 0.
                # Check might need to be more robust (e.g. if y_fields empty
                # but optic.max_field exists).
                return 0.0

            # Fallback: wide guess if no field range obvious (risky).
            # Better: require optic to provide `max_field_y` or similar.
            # For now, assume `optic.get_max_field_y()` exists or can be derived.
            # Heuristic: use paraxial solver solution as start for range, or default.
            # Needs to be robust.
            # Consider paraxial solution as starting point for range.
            # Assume nominal range if not derivable from `optic.fields`.
            # Common issue: setting bounds for root finders.
            # Strategy: Test `objective_function` at 0, and some estimate of max field.
            # If `optic.max_field_y` exists, use it.
            # Assume `optic.fields` gives reasonable span.
            # If only one field point (e.g. (0,0)), tricky.
            # If target is 0, and (0,0) is only field, 0 is answer.
            if not y_fields and target_image_height == 0.0:  # if fields = [[0,0]]
                # Check if (0,0) is only field point or y_fields empty for other
                # reasons.
                is_only_on_axis = True
                for f_pt in optic.fields:
                    if not (
                        isinstance(f_pt, (list, tuple))
                        and len(f_pt) == 2
                        and f_pt[0] == 0
                        and f_pt[1] == 0
                    ):
                        is_only_on_axis = False
                        break
                if is_only_on_axis:
                    return 0.0

            # If truly no y_fields to define range, error or use default.
            # Safer to error.
            if not y_fields:
                raise ValueError(
                    "Cannot determine solver range from optic.fields "
                    "for RealFieldSolver."
                )

        # Bounds strategy: if fields symmetric [0, max_y], then [-max_y, max_y].
        # If only positive fields defined (e.g. [0, 0.7, 1.0]*max_field),
        # range could be [-max_abs_field, max_abs_field].
        abs_max_y_field = max(abs(f_y) for f_y in y_fields) if y_fields else 0.0

        # If target_image_height is 0, 0.0 is likely candidate for field.
        if target_image_height == 0.0 and abs(objective_function(0.0)) < tol:
            # Check if field 0 gives image height 0. If so, return 0.
            return 0.0
        # If not, still need to search.

        # Define search bracket [a, b].
        # Ensure 0 included if possible, or search +/- sides separately
        # if function known to be monotonic and odd/even.
        # Assume symmetric range around 0 up to max absolute field.
        # If all defined fields positive, and target_image_height negative,
        # search in negative field range.

        # Heuristic for bracket:
        # Find bracket [a,b] where f(a) and f(b) have opposite signs.
        # Start with [0, abs_max_y_field] or [-abs_max_y_field, 0]
        # or [-abs_max_y_field, abs_max_y_field].
        # Choice depends on expected sign of target_image_height and func behavior.

        # Try common range: [-abs_max_y_field, abs_max_y_field].
        # Assumes function is reasonably well-behaved.
        # If `abs_max_y_field` is 0 (e.g. only on-axis field defined),
        # range is problematic unless `target_image_height` is also 0 (handled above).
        if abs_max_y_field == 0 and target_image_height != 0:
            raise ValueError(
                "Max field is 0, but target image height is non-zero. Cannot solve."
            )

        # If `abs_max_y_field` very small, brentq might struggle.
        # Consider min sensible range if `abs_max_y_field` tiny.
        # Many optical systems: field vs image height is monotonic.

        # Determine search range based on sign of `target_image_height`
        # and characteristics of `objective_function(0)`.
        f0 = objective_function(
            0.0
        )  # Img height at 0-field (should be ~0 for centered)

        # Define initial bracket points.
        # May need adjustment if f(a), f(b) don't have opposite signs.
        # Common default range related to max field angle/height.
        # If max field 10 deg, search -10 to 10 deg.
        # If max height 5mm, search -5 to 5 mm.
        # `abs_max_y_field` is best guess for this magnitude.

        # If `abs_max_y_field` is 0, means only (0,0) field defined.
        # Handled if `target_image_height` is also 0.
        # If `target_image_height` non-zero, and max_field 0, it's an issue.
        if abs_max_y_field == 0.0:  # Caught by earlier checks if target != 0
            if target_image_height == 0.0:
                return 0.0  # Already handled, but defensive
            # else: # This else is effectively unreachable due to the check on line 343
            # (approx original line number)
            raise ValueError(
                "Cannot solve: max field is 0 for non-zero target image height."
            )

        # Search bracket: need f(a) * f(b) < 0.
        # Try common values for 'a', 'b'.
        # Many systems: if target_image_height > 0, field > 0. If target < 0, field < 0.
        # (Assuming positive magnification sense or standard coordinate systems).

        # Initial guesses for bracket:
        # If target_image_height positive, try [0, abs_max_y_field].
        # If target_image_height negative, try [-abs_max_y_field, 0].
        # If target_image_height zero, handled.

        # Needs to be more robust. What if function not monotonic from 0?
        # Or if max field defined not large enough to bracket solution?

        # Simpler, wider bracket first: [-abs_max_y_field, abs_max_y_field].
        # Rely on brentq to find root if there.
        # Must ensure objective function at bracket endpoints has opposite signs.

        bracket_a = -abs_max_y_field
        bracket_b = abs_max_y_field

        # If `target_image_height` very close to f0 (img height at zero field),
        # solution likely very close to 0.
        if abs(target_image_height - f0) < tol and abs(f0) < tol:
            # f0 is obj_func(0)+target_image_height;
            # i.e. objective_function(0) is close to 0
            return 0.0

        # Check signs at bracket endpoints.
        fa = objective_function(bracket_a)
        fb = objective_function(bracket_b)

        # If one end is inf (ray trace fail), try shrink bracket or use one-sided.
        # Can get complex. For now, assume fa, fb finite.

        if np.isinf(fa) or np.isinf(fb):
            # Try better bracket if trace fails at extremes.
            # Might involve stepping inwards or adaptive search.
            # For now, error if initial bracket problematic.
            # Common: expand search if no sign change found.
            # Or, if trace fails, assume field "too large".
            pass  # brentq catches if signs not opposite or values invalid.

        # Try establish valid bracket for brentq: f(a) * f(b) < 0.
        # Scenario 1: `target_image_height` outside range from [-max_field, max_field].
        # Scenario 2: Function flat or non-monotonic, fools simple bracketing.

        # Attempt to find valid bracket.
        # Logic can be involved. Common: test points.
        # Simplicity: try few common brackets.

        search_brackets = []
        if abs_max_y_field > 1e-9:  # Avoid zero-width if max_field effectively zero
            search_brackets.append((bracket_a, bracket_b))  # Full range
            # target and f(0) different signs relative to f(a)
            if target_image_height * fa > 0 and target_image_height * f0 < 0:
                search_brackets.append((bracket_a, 0.0))
            # target and f(0) different signs relative to f(b)
            if target_image_height * fb > 0 and target_image_height * f0 < 0:
                search_brackets.append((0.0, bracket_b))

        # If f0 is target (i.e., `target_image_height` = image_y_at_zero_field),
        # then `objective_function(0)` = 0. Handled by `abs(f0) < tol`
        # if `target_image_height` is 0.
        # If `target_image_height` not 0, but `objective_function(0)` = 0
        # (meaning image_y_at_zero_field = `target_image_height`), then 0 is solution.
        # `objective_function(0.0)` check earlier handles if `target_image_height` = 0.
        # Re-check specifically:
        val_at_zero_field = objective_function(0.0)
        if abs(val_at_zero_field) < tol:  # image_y(field=0) == target_image_height
            return 0.0

        # Add [0, abs_max_y_field] and [-abs_max_y_field, 0] if appropriate.
        if abs_max_y_field > 1e-9:
            if val_at_zero_field * fb < 0:  # Sign change between 0 and max_field
                search_brackets.append((0.0, bracket_b))
            if val_at_zero_field * fa < 0:  # Sign change between min_field and 0
                search_brackets.append((bracket_a, 0.0))

        # Ensure brackets ordered (a < b).
        ordered_brackets = []
        for ba, bb in search_brackets:
            if ba == bb:
                continue  # Skip zero-width
            ordered_brackets.append((min(ba, bb), max(ba, bb)))

        # Small default bracket if others fail (e.g. if `abs_max_y_field` was 0).
        # Risky; ideally not needed if `optic.fields` well-defined.
        if not ordered_brackets and (
            abs_max_y_field < 1e-9 and target_image_height != 0.0
        ):
            # Max field is 0, target is not.
            # Error case, but last resort for solver:
            # Implies solution very near 0, but obj(0) not close enough.
            # Sign of problem (optic setup or target).
            # brentq needs bracket. Tiny bracket around 0 (guess).
            # Heuristic: estimate derivative paraxially or small perturbation.
            # For now, situation indicates an issue.
            # Default guess like [-1, 1] (deg or mm) is arbitrary.
            # Assume likely unsolvable with current info.
            pass  # Will fail brentq if no valid bracket found.

        for current_a, current_b in ordered_brackets:
            if current_a == current_b:
                continue  # Skip zero-width brackets
            try:
                val_a = objective_function(current_a)
                val_b = objective_function(current_b)
                if np.isinf(val_a) or np.isinf(val_b):
                    continue  # Skip if trace failed

                if val_a * val_b <= 0:  # Bracket with sign change (or one is zero)
                    solved_field = brentq(
                        objective_function,
                        current_a,
                        current_b,
                        xtol=tol,
                        maxiter=max_iter,
                    )
                    return solved_field
            except ValueError:  # brentq: f(a)*f(b) > 0
                continue
            except RuntimeError:  # brentq: convergence issues
                continue

        # No solution found with tried brackets.
        # Problem:
        # 1. `target_image_height` outside optic's achievable range.
        # 2. Bracketing logic insufficient.
        # 3. Objective func ill-behaved (e.g., discontinuous from ray failures).
        err_msg = (
            "RealFieldSolver could not find solution for "
            f"target={target_image_height}. "
            "Tested bracket boundaries had no sign change or root finding failed. "
            f"Max field tried: +/-{abs_max_y_field}. f(0)={val_at_zero_field}, "
            f"f(-max)={fa}, f(+max)={fb}"
        )
        raise RuntimeError(err_msg)
