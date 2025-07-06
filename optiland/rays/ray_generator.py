"""Ray Generator Module.

This module contains the RayGenerator class, which is used to generate rays
for tracing through an optical system.

Kramer Harrison, 2024
"""

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays


class RayGenerator:
    """Generator class for creating rays."""

    def __init__(self, optic):
        """Initialize RayGenerator.

        Args:
            optic (Optic): The optical system instance.

        """
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """Generate rays for tracing based on the given parameters.

        Args:
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            wavelength (float): Wavelength of the rays.

        Returns:
            RealRays: RealRays object containing the generated rays.

        """
        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self.optic.field_type.get_ray_origins(
            self.optic, Hx, Hy, Px, Py, vx, vy
        )

        if self.optic.obj_space_telecentric:
            # Validations for obj_space_telecentric combined with specific
            # field types or aperture types are now handled by the
            # field_type strategy's `validate_optic_state` method.
            # This method is called when `optic.set_field_type()` is invoked.
            # Thus, incompatible configurations should be caught at setup time.

            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        apodization = self.optic.apodization
        if apodization:
            intensity = apodization.get_intensity(Px, Py)
        else:
            intensity = be.ones_like(Px)

        wavelength_arr = be.ones_like(x1) * wavelength

        if self.optic.polarization == "ignore":
            if self.optic.surface_group.uses_polarization:
                raise ValueError(
                    "Polarization must be set when surfaces have "
                    "polarization-dependent coatings."
                )
            return RealRays(x0, y0, z0, L, M, N, intensity, wavelength_arr)
        return PolarizedRays(x0, y0, z0, L, M, N, intensity, wavelength_arr)

    # _get_ray_origins logic is now in field strategy classes.
    # The _get_starting_z_offset logic was replicated in AngleField strategy.

    # def _get_starting_z_offset(self): # Logic moved to AngleField strategy
    #     """Calculate the starting ray z-coordinate offset for systems with an
    #     object at infinity. This is relative to the first surface of the optic.
    #
    #     This method chooses a starting point that is equivalent to the entrance
    #     pupil diameter of the optic.
    #
    #     Returns:
    #         float: The z-coordinate offset relative to the first surface.
    #
    #     """
    #     z = self.optic.surface_group.positions[1:-1]
    #     offset = self.optic.paraxial.EPD()
    #     return offset - be.min(z)
