"""Paraxial Ray Tracer Module

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
        if be.any(be.isclose(delta_z, 0.0)):
            if self.optic.object_surface.is_infinite:
                # For an infinite object, u0 is typically 0 for rays starting
                # parallel to the axis (e.g., marginal ray definition).
                u0 = be.zeros_like(y0)
            else:  # Finite object at EPL
                # If a finite object is at EPL (EPL == z0), then y0 should be ~y1.
                # The formula (y1-y0)/(EPL-z0) becomes 0/0, making u0 ill-defined.
                raise ValueError(
                    "Object is at EPL; paraxial ray angle u0 is ill-defined."
                )
        else:  # Standard case: EPL != z0
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

        for k_idx, surf_k in enumerate(surfs):
            if k_idx < skip:  # Skip surfaces if requested
                continue

            if isinstance(surf_k, ObjectSurface):
                heights.append(be.copy(y))
                slopes.append(be.copy(u))
                continue

            # Propagate to surface k
            t = pos[k_idx] - z  # Thickness to propagate
            z = pos[k_idx]  # Update z to current surface's global position
            y = y + t * u  # Ray height at surface k

            # Refract or Reflect at surface k
            if surf_k.is_reflective:
                u = -u - 2 * y / R[k_idx]
            else:
                u = (1 / n_indices[k_idx]) * (
                    n_indices[k_idx - 1] * u - y * power[k_idx]
                )

            heights.append(be.copy(y))
            slopes.append(be.copy(u))

            if k_idx >= len(R) - 1 + skip - (len(surfs) - len(R)):
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
            return be.asarray(x)
