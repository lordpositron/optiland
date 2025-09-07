from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial


@dataclass
class Layer:
    """Represents a thin-film layer.

    Parameters
    ----------
    material : BaseMaterial
        Optiland material providing ``n(wavelength)`` and ``k(wavelength)``.
    thickness_um : float
        Layer thickness in microns (µm).
    name : str | None
        Optional label for display.

    Examples
    --------
    >>> from optiland.materials import IdealMaterial
    >>> from optiland.thin_film import Layer
    >>> sio2 = IdealMaterial(1.46)
    >>> layer = Layer(sio2, thickness_um=0.1, name="SiO2 100 nm")
    """

    material: BaseMaterial
    thickness_um: float
    name: str | None = None

    def n_complex(self, wavelength_um):
        """Complex index n~ = n + i k for one or multiple wavelengths."""
        n = self.material.n(wavelength_um)
        k = self.material.k(wavelength_um)
        # S'assurer que n et k sont des arrays broadcastables
        n = be.atleast_1d(n)
        k = be.atleast_1d(k)
        return be._lib.asarray(n, dtype=be._lib.complex128) + 1j * be._lib.asarray(
            k, dtype=be._lib.complex128
        )

    def phase_thickness(
        self,
        wavelength_um: float | be.ndarray,
        cos_theta_l: complex | be.ndarray,
        n_complex_l: complex | be.ndarray,
    ) -> complex | be.ndarray:
        """Phase δ = 2π/λ·n·d·cos(θ_l)

        Inputs must be broadcastable over wavelength and AOI grids.
        """
        k0 = 2 * be.pi / wavelength_um  # µm^-1
        return k0 * n_complex_l * self.thickness_um * cos_theta_l
