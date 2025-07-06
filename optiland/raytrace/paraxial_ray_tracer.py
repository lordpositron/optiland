"""Paraxial Ray Tracer Module.

This module contains the ParaxialRayTracer class, which is responsible for
tracing paraxial rays through an optical system.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.surfaces import ObjectSurface


class ParaxialRayTracer:
    """Class to trace paraxial rays through an optical system.

    Args:
        optic (Optic): The optical system to be traced.

    """

    def __init__(self, optic):
        """Initialize ParaxialRayTracer.

        Args:
            optic (Optic): The optical system instance.

        """
        self.optic = optic

    def trace(self, Hy, Py, wavelength):
        """Trace a paraxial ray using normalized field and pupil coordinates.

        Args:
            Hy (float): Normalized field coordinate.
            Py (float): Normalized pupil coordinate.
            wavelength (float): Wavelength of the light.

        """
        EPL = self.optic.paraxial.EPL()
        EPD = self.optic.paraxial.EPD()

        y1 = Py * EPD / 2  # Height at the entrance pupil for this pupil coordinate

        if not self.optic.field_type:
            raise RuntimeError("Optic.field_type strategy is not set.")
        y0, z0 = self.optic.field_type.get_paraxial_object_position(
            self.optic, Hy, y1, EPL
        )

        # Calculate initial ray angle u0.
        # This involves handling potential division by zero if the object is
        # at the entrance pupil location (EPL).
        delta_z = EPL - z0
        if be.isclose(delta_z, 0.0).any():
            if self.optic.object_surface.is_infinite:
                # For an infinite object, u0 is typically 0 for rays starting
                # parallel to the axis (e.g., marginal ray definition).
                # Strategies provide appropriate y0, z0 for this.
                u0 = be.zeros_like(y0)
            else:  # Finite object at EPL
                # If a finite object is at EPL (EPL == z0), then y0 should be ~y1.
                # The formula (y1-y0)/(EPL-z0) becomes 0/0, making u0 ill-defined.
                raise ValueError(
                    "Object is at EPL; paraxial ray angle u0 is ill-defined."
                )
        else: # Standard case: EPL != z0
            u0 = (y1 - y0) / delta_z

        rays = ParaxialRays(y0, u0, z0, wavelength)

        self.optic.surface_group.trace(rays)

    def trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        """Trace generically-defined paraxial rays through the optical system.

        Args:
            y (float | be.ndarray): The initial height(s) of the rays.
            u (float | be.ndarray): The initial slope(s) of the rays.
            z (float | be.ndarray): The initial axial position(s) of the rays.
            wavelength (float): The wavelength of the rays.
            reverse (bool, optional): If True, trace the rays in reverse
                direction. Defaults to False.
            skip (int, optional): The number of surfaces to skip during
                tracing. Defaults to 0.

        Returns:
            tuple: A tuple containing the final height(s) and slope(s) of the
                rays after tracing.

        """
        y = self._process_input(y)
        u = self._process_input(u)
        z = self._process_input(z)

        R = self.optic.surface_group.radii
        n_indices = self.optic.n(wavelength)
        pos = be.ravel(self.optic.surface_group.positions)
        surfs = self.optic.surface_group.surfaces

        if reverse:
            R = -be.flip(R)
            n_indices = be.roll(n_indices, shift=1)
            n_indices = be.flip(n_indices)
            pos = pos[-1] - be.flip(pos)
            surfs = surfs[::-1]

        power = be.diff(n_indices, prepend=be.array([n_indices[0]])) / R

        heights = []
        slopes = []

        for k_idx, surf_k in enumerate(surfs): # Use enumerate for clarity if needed
            if k_idx < skip: # Skip surfaces if requested
                # For skipped surfaces, we might still need to record initial y, u
                # if the output format expects values for all surfaces.
                # However, typical paraxial traces only care about values *after* interaction.
                # The original code's loop `range(skip, len(R))` implies we just don't
                # process these. If output must align with all surfaces, this needs adjustment.
                # For now, assume skip means these surfaces are entirely ignored.
                continue # This was `for k in range(skip, len(R))`

            if isinstance(surf_k, ObjectSurface):
                heights.append(be.copy(y))
                slopes.append(be.copy(u))
                continue

            # Propagate to surface k
            # Ensure pos indexing matches R, power, and n_indices after potential reverse
            # If R has N elements (0 to N-1), pos should also correspond.
            # k_idx here is 0-based index into potentially reversed 'surfs' array.
            # We need thickness t = pos[k_idx] - z_prev_surface_global
            # z is current_z_global_coord_of_ray_start_for_this_segment

            t = pos[k_idx] - z # Thickness to propagate
            z = pos[k_idx]     # Update z to current surface's global position
            y = y + t * u      # Ray height at surface k

            # Refract or Reflect at surface k
            if surf_k.is_reflective:
                # Assuming R[k_idx] is the radius of the k-th surface in the list
                # n_indices[k_idx] is n', n_indices[k_idx-1] is n (for refraction)
                # For reflection, n' = -n. The paraxial formulas usually handle this
                # by specific reflection equations.
                # Paraxial reflection: u' = u - 2*y/R (if R is positive for convex from left)
                # Or u_reflected = -u_incident - 2*y/R_mirror (sign conventions vary)
                # Optiland's convention: R > 0 for center to right.
                # If light from left, hits convex (R>0), reflected u should be smaller if y>0.
                # u_inc = u; u_refl = u_inc - 2*y*n_inc/R (No, this is for OPL change)
                # Standard paraxial reflection: u_after = u_before - 2*y/R (if n=1 before mirror)
                # Or, more generally, if using n' = -n:
                # n*u = n_prev*u_prev - y * (n-n_prev)/R_surf
                # For mirror, n_after = -n_before.
                # -n_before*u_after = n_before*u_before - y*(-n_before - n_before)/R_surf
                # -u_after = u_before - y*(-2)/R_surf  => u_after = -u_before - 2*y/R_surf
                u = -u - 2 * y / R[k_idx]
            else:
                # Paraxial refraction: n_k * u_k = n_{k-1} * u_{k-1} - y_k * (n_k - n_{k-1}) / R_k
                # u_k = (1/n_k) * (n_{k-1}*u_{k-1} - y_k * power_k)
                # Here, u is u_{k-1} (slope before surface k). power[k_idx] is (n_k - n_{k-1})/R_k.
                # n_indices[k_idx] is n_k, n_indices[k_idx-1] is n_{k-1} (after reversing if any)
                u = (1 / n_indices[k_idx]) * (
                    n_indices[k_idx -1] * u - y * power[k_idx]
                )

            heights.append(be.copy(y))
            slopes.append(be.copy(u))

            if k_idx >= len(R) -1 + skip - (len(surfs) - len(R)): # Adjust loop if R, surfs differ post skip
                 break


        heights_arr = be.array(heights)
        slopes_arr = be.array(slopes)

        # Ensure output shape is consistent, e.g., (num_surfaces_traced, num_rays)
        # If y,u were scalars, reshape to (len, 1)
        if heights_arr.ndim == 1:
            heights_arr = heights_arr.reshape(-1, 1)
        if slopes_arr.ndim == 1:
            slopes_arr = slopes_arr.reshape(-1, 1)

        return heights_arr, slopes_arr

    # _get_object_position logic is now in field strategy classes.

    def _process_input(self, x):
        """Process input to ensure it is a numpy array.

        Args:
            x (float | int | list | be.ndarray): The input to process.

        Returns:
            be.ndarray: The processed input as a backend array.

        """
        if isinstance(x, (int, float)):
            return be.array([x])
        else:
            return be.asarray(x) # Use asarray for existing backend arrays
