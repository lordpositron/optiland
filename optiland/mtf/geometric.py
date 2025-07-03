"""Geometric Modulation Transfer Function (MTF) Module.

This module provides the GeometricMTF class for computing the MTF
of an optical system based on its geometric Point Spread Function (PSF).

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.psf.geometric import GeometricPSF

from .base import BaseMTF


class GeometricMTF(BaseMTF):
    """Geometric Modulation Transfer Function (MTF) class.

    This class calculates and visualizes the Modulation Transfer Function (MTF)
    of an optic using a geometric PSF. The geometric PSF is essentially a
    normalized 2D histogram of ray intersection points (spot diagram).
    The MTF is then derived from this PSF.

    The calculation method is based on the principle that the MTF is related
    to the Fourier transform of the Line Spread Function (LSF), and the LSF
    can be obtained by projecting the PSF.
    See: Smith, Modern Optical Engineering 3rd edition, Section 11.9 for the
    underlying principles of geometric MTF from spot data.

    Args:
        optic (Optic): The optic for which to calculate the MTF.
        fields (str or list, optional): The field coordinates for which to
            calculate the MTF. Defaults to 'all'.
        wavelength (str or float, optional): The wavelength of light to use
            for the MTF calculation. Defaults to 'primary'.
        num_rays (int, optional): The number of rays to trace for the
            underlying PSF's spot diagram. Defaults to 1000.
        distribution (str, optional): The distribution of rays for the spot
            diagram. Defaults to 'uniform'.
        psf_bins (int, optional): The number of bins along each axis for the
            2D histogram that forms the PSF. Defaults to 128.
        num_points (int, optional): The number of points to sample in the MTF
            curve. Defaults to 256.
        max_freq (str or float, optional): The maximum frequency to consider
            in the MTF curve (cycles/mm). Defaults to 'cutoff', which calculates
            it based on wavelength and F-number.
        scale (bool, optional): Whether to scale the MTF curve by the
            diffraction-limited MTF. Defaults to True.

    Attributes:
        num_rays (int): Number of rays for PSF.
        distribution (str): Ray distribution for PSF.
        psf_bins (int): Number of bins for PSF histogram.
        num_points (int): Number of points in the MTF curve.
        scale (bool): Whether to scale MTF by diffraction limit.
        max_freq (float): Maximum frequency for MTF plot.
        psf_results (list[GeometricPSF]): List of GeometricPSF objects,
            one for each field.
        mtf (list): List of MTF data. Each item is a list [tangential_mtf,
            sagittal_mtf] for a field.
        freq (be.ndarray): Array of frequency points for the MTF curve.
        diffraction_limited_mtf (be.ndarray): The diffraction-limited MTF curve,
            used if `scale` is True.
    """

    def __init__(
        self,
        optic,
        fields="all",
        wavelength="primary",
        num_rays=1000,
        distribution="uniform",
        psf_bins=128,
        num_points=256,
        max_freq="cutoff",
        scale=True,
    ):
        self.num_rays = num_rays
        self.distribution = distribution
        self.psf_bins = psf_bins
        self.num_points = num_points
        self.scale = scale

        super().__init__(optic, fields, wavelength)

        if max_freq == "cutoff":
            fno = self.optic.paraxial.FNO()
            if fno == 0:
                self.max_freq = 100  # Default fallback
            else:
                self.max_freq = 1 / (self.resolved_wavelength * 1e-3 * fno)
        else:
            self.max_freq = max_freq

        self.freq = be.linspace(0, self.max_freq, self.num_points)

        if self.scale:
            safe_max_freq = self.max_freq if self.max_freq > 0 else 1.0
            ratio = be.clip(self.freq / safe_max_freq, 0.0, 1.0)
            phi = be.arccos(ratio)
            self.diffraction_limited_mtf = 2 / be.pi * (phi - be.cos(phi) * be.sin(phi))
        else:
            self.diffraction_limited_mtf = be.ones_like(self.freq)

    def _calculate_psf(self):
        """Calculates and stores the GeometricPSF for each field.

        This method is called by the `BaseMTF` constructor.
        It populates `self.psf_results`.
        """
        self.psf_results = []
        for field_coord in self.resolved_fields:
            psf_instance = GeometricPSF(
                optic=self.optic,
                field=field_coord,
                wavelength=self.resolved_wavelength,
                num_rays=self.num_rays,
                distribution=self.distribution,
                bins=self.psf_bins,
                normalize=True,
            )
            self.psf_results.append(psf_instance)

    def _generate_mtf_data(self):
        """Generates the MTF data from the calculated PSFs.

        This method is called by the `BaseMTF` constructor after `_calculate_psf`.

        Returns:
            list: A list of MTF data for each field. Each item is a list
                  [tangential_mtf, sagittal_mtf].
        """
        mtf_all_fields = []
        for psf_instance in self.psf_results:
            psf_image = psf_instance.psf
            x_edges = psf_instance.x_edges
            y_edges = psf_instance.y_edges

            # LSFy (for tangential MTF) is projection of PSF onto y-axis
            # LSFx (for sagittal MTF) is projection of PSF onto x-axis
            # PSF shape is (num_y_bins, num_x_bins)
            lsf_y = be.sum(psf_image, axis=1)  # Sum over x-bins
            lsf_x = be.sum(psf_image, axis=0)  # Sum over y-bins

            # Coordinates for LSFs (bin centers)
            # y_coords are for lsf_y, x_coords are for lsf_x
            y_coords = (y_edges[:-1] + y_edges[1:]) / 2
            x_coords = (x_edges[:-1] + x_edges[1:]) / 2

            # Ensure coordinates are sorted for _compute_mtf_from_lsf
            sort_idx_y = be.argsort(y_coords)
            y_coords = y_coords[sort_idx_y]
            lsf_y = lsf_y[sort_idx_y]

            sort_idx_x = be.argsort(x_coords)
            x_coords = x_coords[sort_idx_x]
            lsf_x = lsf_x[sort_idx_x]

            # Tangential MTF from LSFy (distribution along y)
            mtf_tangential = self._compute_mtf_from_lsf(lsf_y, y_coords, self.freq)
            # Sagittal MTF from LSFx (distribution along x)
            mtf_sagittal = self._compute_mtf_from_lsf(lsf_x, x_coords, self.freq)

            if self.scale:
                mtf_tangential *= self.diffraction_limited_mtf
                mtf_sagittal *= self.diffraction_limited_mtf

            mtf_all_fields.append([mtf_tangential, mtf_sagittal])
        return mtf_all_fields

    def _compute_mtf_from_lsf(self, lsf, coords, frequencies):
        """Computes MTF from a Line Spread Function (LSF).

        Args:
            lsf (be.ndarray): The Line Spread Function values.
            coords (be.ndarray): The spatial coordinates corresponding to LSF values.
                These should be in micrometers (µm).
            frequencies (be.ndarray): The spatial frequencies (cycles/mm) at
                which to compute MTF.

        Returns:
            be.ndarray: The MTF values corresponding to the input frequencies.
        """
        # Ensure LSF is normalized (sum of LSF * dx should be 1)
        # coords are bin centers. dx is spacing between them.
        if len(coords) < 2:  # Not enough points for dx
            return be.zeros_like(frequencies)

        dx = be.abs(coords[1] - coords[0])  # Assuming uniform spacing

        # Normalize LSF such that sum(lsf * dx) = 1 for it to be a probability density
        lsf_sum_dx = be.sum(lsf * dx)
        lsf_normalized = lsf / lsf_sum_dx if lsf_sum_dx > 1e-9 else lsf

        mtf_values = be.zeros_like(frequencies)

        # Convert coords from µm to mm for consistency with frequencies (cycles/mm)
        coords_mm = coords * 1e-3

        for i, freq_val in enumerate(frequencies):
            # MTF is the modulus of the Fourier Transform of the LSF
            # MTF(ν) = |∫ LSF(x) * exp(-j * 2π * ν * x) dx|
            # Discrete form: |Σ LSF(x_k) * exp(-j * 2π * ν * x_k) * Δx|

            # Compute cos and sin terms separately (real/imag parts of FT)
            # Ac = Σ LSF(x_k) * cos(2π * ν * x_k) * Δx
            # As = Σ LSF(x_k) * sin(2π * ν * x_k) * Δx
            # MTF = sqrt(Ac^2 + As^2)

            cos_term = be.cos(2 * be.pi * freq_val * coords_mm)
            sin_term = be.sin(2 * be.pi * freq_val * coords_mm)

            # Use normalized LSF for Ac and As
            ac_val = be.sum(lsf_normalized * cos_term * dx)
            as_val = be.sum(lsf_normalized * sin_term * dx)

            mtf_values[i] = be.sqrt(ac_val**2 + as_val**2)

        if mtf_values[0] > 1e-9:  # Avoid division by zero
            mtf_values = mtf_values / mtf_values[0]

        return mtf_values

    def _plot_field_mtf(self, ax, field_index, mtf_field_data, color):
        """Plots the MTF data for a single field. Called by BaseMTF.view().

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axes object.
            field_index (int): The index of the current field in `self.resolved_fields`.
            mtf_field_data (list): A list containing [tangential_mtf, sagittal_mtf]
                                   (be.ndarray) for the field.
            color (str): The color to use for plotting this field.
        """
        current_field_label_info = self.resolved_fields[field_index]
        freq_np = be.to_numpy(self.freq)

        # Plot tangential MTF
        ax.plot(
            freq_np,
            be.to_numpy(mtf_field_data[0]),  # Tangential data
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Tangential"
            ),
            color=color,
            linestyle="-",
        )
        # Plot sagittal MTF
        ax.plot(
            freq_np,
            be.to_numpy(mtf_field_data[1]),  # Sagittal data
            label=(
                f"Hx: {current_field_label_info[0]:.1f}, "
                f"Hy: {current_field_label_info[1]:.1f}, Sagittal"
            ),
            color=color,
            linestyle="--",
        )
