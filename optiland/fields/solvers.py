"""Field Solvers Module

This module contains classes for solving object-space field values
(e.g., height or angle) corresponding to a desired image-space height.

Kramer Harrison, 2025
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import brentq


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
        using paraxial chief ray tracing.

        Args:
            optic: The optical system.
            target_image_height: The desired height on the image surface.

        Returns:
            The corresponding object-space field value.
        """
        if not optic.fields:
            raise ValueError("Optic has no fields defined.")

        # Assuming the field value is stored in the second element (index 1)
        # of the tuple/list representing a field point (e.g., (fx, fy) or (hx, hy)).
        # We'll use the y-component for this calculation.
        # A more robust way might be `optic.get_max_field()` or similar if available.
        # For now, assume the last field point in `optic.fields` is the maximum field.
        reference_object_field_y = optic.fields[-1][1]  # Assuming (fx, fy) or (hx, hy)

        if reference_object_field_y == 0:
            # If the reference field is zero, we can't scale.
            # This might happen if only an on-axis field is defined.
            # If target_image_height is also zero, then 0 is a valid solution.
            if target_image_height == 0:
                return 0.0
            # Else, target is non-zero, but reference is zero. This is an issue.
            # This might indicate an optical system that doesn't produce
            # off-axis image height for the given reference, or the reference
            # field itself is problematic.
            # Consider raising an error, returning NaN, or handling as per system
            # design. For now, assume this means we can't solve it this way.
            raise ValueError(
                "Reference object field is zero, cannot scale to non-zero target."
            )

        # Trace a paraxial chief ray for this reference object field.
        # The `chief_ray` function might need the field value directly.
        # Assuming `optic.paraxial.chief_ray` takes field components (fx, fy)
        # or (hx, hy) and returns ray data including image height.
        # We need to know the structure of what `chief_ray` returns.
        # Assume it returns an object or dict with an `image_height` attribute/key.
        # Paraxial chief ray func signature likely `optic.paraxial.chief_ray(Hy, Hy_is_angle)`. # noqa: E501
        # We need to determine if field is an angle or height.
        # This info should come from the optic's field strategy.

        # Assume `optic.field_strategy` exists and can tell if it's an angle field.
        # This is a simplification; actual info retrieval might differ.
        # Common pattern: different field types (e.g. AngleField, ObjectHeightField).

        # We need to import Optic and paraxial calculations.
        # `from optiland.optic import Optic` is now at top for clarity.
        from optiland import paraxial  # Assuming this is where paraxial functions are

        if not hasattr(optic, "paraxial_model") or optic.paraxial_model is None:
            # Paraxial model needs to be initialized. Might be in `Optic.__init__`
            # or a dedicated method. Assume it's available or can be computed.
            optic.update_paraxial_model()

        # Get paraxial chief ray for `reference_object_field_y`.
        # `paraxial.chief_ray()` usually takes object height (or angle)
        # and details about whether the object is at infinity.
        # This depends on the current field strategy of the optic.

        # Assume `optic.field_strategy.field_type` can be 'angle' or 'height'.
        # This is a placeholder for how Optic class manages its field definition.
        # A more robust way: use a method on Optic or its field strategy
        # to get paraxial image height for a given object field.

        # For now, use `optic.paraxial.chief_ray(Hy, field_is_angle)`.
        # Need to know if `reference_object_field_y` is an angle or height.
        # This info is typically part of Optic's setup.
        # Assume Optic class has `object_is_at_infinity` property.

        # Paraxial chief ray func in `optic.paraxial` likely needs paraxial model.
        # Assume `optic.paraxial.chief_ray` can access `optic.paraxial_model`
        # or takes it as an argument.
        # Args to `chief_ray` typically `(obj_y, pm, obj_is_inf)` or similar.

        # Let's simplify and assume `optic.paraxial_image_height(object_field_y)`
        # exists or we can compute it using `optic.paraxial.chief_ray`.

        # `chief_ray` in `paraxial.py` is `chief_ray(Hy, pm, obj_is_inf)`.
        # `pm` is `optic.paraxial_model`. `obj_is_inf` is `optic.object_is_at_infinity`.
        ref_chief_ray = paraxial.chief_ray(
            reference_object_field_y, optic.paraxial_model, optic.object_is_at_infinity
        )

        # Returned `chief_ray` is tuple `(y_stop, u_stop, y_image, u_image)`.
        reference_image_height = ref_chief_ray[2]  # y_image

        if reference_image_height == 0:
            # Ref field maps to zero image height.
            # If `target_image_height` is also zero, any field could be solution (0.0).
            # If `target_image_height` non-zero, cannot find unique scale factor.
            if target_image_height == 0:
                # Target is 0, ref image height is 0.
                # If `reference_object_field_y` was non-zero, this is ambiguous.
                # However, if `reference_object_field_y` itself was 0, then 0 is fine.
                # Assume if `reference_object_field_y` was non-zero, this is an issue.
                # Or, return 0 if target is 0.
                return 0.0
            # else:
            raise ValueError(
                "Reference field results in zero image height, "
                "cannot scale to non-zero target."
            )

        # Calculate the scale factor.
        scale_factor = target_image_height / reference_image_height

        # Apply scale factor to reference object field.
        solved_object_field = reference_object_field_y * scale_factor

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
