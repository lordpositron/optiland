"""Paraxial Ray Tracer Module

This module contains the ParaxialRayTracer class, which is responsible for tracing
paraxial rays through an optical system.

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.surfaces import ObjectSurface


class ParaxialRayTracer:
    """Class to trace paraxial rays through an optical system

    Args:
        optic (Optic): The optical system to be traced.
    """

    def __init__(self, optic):
        self.optic = optic

    def trace(self, Hy, Py, wavelength):
        """Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        The initial ray parameters (y0, u0, z0) are determined by the
        optic's current field group.

        Args:
            Hy (float): Normalized field coordinate, interpreted by the active
                        field group (e.g., normalized angle, object height,
                        or image height).
            Py (float): Normalized pupil coordinate.
            wavelength (float): Wavelength of the light in micrometers.

        """
        if self.optic.fields is None:
            raise ValueError(
                "Cannot trace paraxial ray: Optic has no field group defined."
            )

        y0, u0 = self.optic.fields.to_paraxial_starting_ray(
            Hy=Hy, Py=Py, wavelength=wavelength, optic=self.optic
        )

        if self.optic.object_surface and self.optic.object_surface.is_infinite:
            # For infinite conjugates, y0, u0 from field group are typically defined
            # with y0 at the first optical surface and u0 as the field angle.
            if not self.optic.surface_group.surfaces:
                raise ValueError(
                    "Optic has no surfaces to define z0 for infinite object."
                )
            z0 = self.optic.surface_group.surfaces[0].geometry.cs.z
        elif self.optic.object_surface:  # Finite object distance
            z0 = self.optic.object_surface.geometry.cs.z
        else:
            raise ValueError("Optic has no object surface defined to determine z0.")

        rays = ParaxialRays(y0, u0, z0, wavelength)
        self.optic.surface_group.trace(rays)
        # The ParaxialRays object 'rays' is updated in-place by surface_group.trace()
        # and contains the traced ray data (e.g., rays.y, rays.u).
        # No explicit return here, but results are in 'rays'.
        # The calling Paraxial.trace method might return rays.y, rays.u.

    def trace_generic(self, y, u, z, wavelength, reverse=False, skip=0):
        """
        Trace generically-defined paraxial rays through the optical system.

        Args:
            y (float or array-like): The initial height(s) of the rays.
            u (float or array-like): The initial slope(s) of the rays.
            z (float or array-like): The initial axial position(s) of the rays.
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
        n = self.optic.n(wavelength)
        pos = be.ravel(self.optic.surface_group.positions)
        surfs = self.optic.surface_group.surfaces

        if reverse:
            R = -be.flip(R)
            n = be.roll(n, shift=1)
            n = be.flip(n)
            pos = pos[-1] - be.flip(pos)
            surfs = surfs[::-1]

        power = be.diff(n, prepend=be.array([n[0]])) / R

        heights = []
        slopes = []

        for k in range(skip, len(R)):
            if isinstance(surfs[k], ObjectSurface):
                heights.append(be.copy(y))
                slopes.append(be.copy(u))
                continue

            # propagate to surface
            t = pos[k] - z
            z = pos[k]
            y = y + t * u

            # reflect or refract
            if surfs[k].is_reflective:
                u = -u - 2 * y / R[k]
            else:
                u = 1 / n[k] * (n[k - 1] * u - y * power[k])

            heights.append(be.copy(y))
            slopes.append(be.copy(u))

        heights = be.array(heights).reshape(-1, 1)
        slopes = be.array(slopes).reshape(-1, 1)

        return heights, slopes

    # _get_object_position method removed as its logic is now encapsulated
    # within the FieldGroup subclasses' to_paraxial_starting_ray method.

    def _process_input(self, x):
        """
        Process input to ensure it is a numpy array.

        Args:
            x (float or array-like): The input to process.

        Returns:
            np.ndarray: The processed input.
        """
        if isinstance(x, (int, float)):
            return be.array([x])
        else:
            return be.array(x)
