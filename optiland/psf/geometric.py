"""Geometric Point Spread Function (PSF) Module.

This module provides functionality for simulating the Point Spread Function (PSF)
of optical systems based on geometric ray tracing data (spot diagrams).

Kramer Harrison, 2025
"""

import optiland.backend as be
from optiland.psf.base import BasePSF


class GeometricPSF(BasePSF):
    """Class representing the Geometric PSF.

    This class computes the PSF of an optical system by generating a 2D
    histogram of ray intersection points on the image plane (a spot diagram).
    It inherits common visualization and initialization functionalities from
    `BasePSF`.

    Args:
        optic (Optic): The optical system object.
        field (tuple): The field point (e.g., (Hx, Hy) in normalized field
            coordinates) at which to compute the PSF.
        wavelength (float): The wavelength of light in micrometers.
        num_rays (int, optional): The number of rays to trace for the spot
            diagram. Defaults to 1000.
        distribution (str, optional): The distribution of rays at the entrance
            pupil. Defaults to 'uniform'.
        bins (int, optional): The number of bins along each axis (x and y) for
            the 2D histogram that forms the PSF. Defaults to 128.
        normalize (bool, optional): Whether to normalize the PSF so that its
            values sum to 1. Defaults to True.

    Attributes:
        psf (be.ndarray): The computed Point Spread Function. This is a 2D
            array representing the intensity distribution in the image plane.
        x_edges (be.ndarray): The bin edges along the x-axis of the PSF.
        y_edges (be.ndarray): The bin edges along the y-axis of the PSF.
        spot_data (SpotDiagram): The underlying spot diagram data used to
            generate the PSF.
    """

    def __init__(
        self,
        optic,
        field,
        wavelength,
        num_rays=1000,
        distribution="uniform",
        bins=128,
        normalize=True,
    ):
        super().__init__(
            optic=optic, field=field, wavelength=wavelength, num_rays=num_rays
        )
        self.distribution = distribution
        self.bins = bins
        self.normalize = normalize

        from optiland.analysis import SpotDiagram  # Moved import

        # Store spot data for potential use in _get_psf_units or other methods
        self.spot_data = SpotDiagram(
            optic=self.optic,
            fields=[self.fields[0]],  # BasePSF stores fields as a list
            wavelengths=[self.wavelengths[0]],  # BasePSF stores wavelengths as a list
            num_rings=self.num_rays,  # Pass num_rays as num_rings
            distribution=self.distribution,
            # coordinates can be left as default 'local' for SpotDiagram
        )

        self.psf, self.x_edges, self.y_edges = self._compute_psf()

    def _compute_psf(self):
        """Computes the PSF from ray data using a 2D histogram.

        This method:
        1. Retrieves ray intersection coordinates (x, y) from the spot diagram
           data for the specified field and wavelength.
        2. Computes a 2D histogram of these coordinates. The number of bins
           is determined by `self.bins`.
        3. Optionally normalizes the histogram so that the sum of its elements
           is 1.
        4. The histogram is then considered the PSF.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]:
                - psf_image (be.ndarray): The computed 2D PSF.
                - x_edges (be.ndarray): Bin edges along the x-axis.
                - y_edges (be.ndarray): Bin edges along the y-axis.
        """
        if not self.spot_data.data or not self.spot_data.data[0]:
            # This case might occur if ray tracing failed or produced no valid rays.
            # Return empty or zero arrays as appropriate.
            psf_image = be.zeros((self.bins, self.bins))
            x_edges = be.linspace(-1, 1, self.bins + 1)  # Default edges
            y_edges = be.linspace(-1, 1, self.bins + 1)  # Default edges
            return psf_image, x_edges, y_edges

        # spot_data.data is a list of lists (one per field, one per wavelength)
        # For GeometricPSF, we have one field and one wavelength.
        spot_field_data = self.spot_data.data[0]
        spot_wavelength_data = spot_field_data[
            0
        ]  # Access the first (and only) wavelength

        x_coords = spot_wavelength_data.x
        y_coords = spot_wavelength_data.y

        # Determine range for histogramming to center the PSF
        # Use percentile to be robust against outliers if any
        x_min, x_max = be.percentile(x_coords, [0.1, 99.9])
        y_min, y_max = be.percentile(y_coords, [0.1, 99.9])

        hist_range = [[x_min, x_max], [y_min, y_max]]

        psf_image, x_edges, y_edges = be.histogram2d(
            x_coords, y_coords, bins=self.bins, range=hist_range
        )

        # Transpose because histogram2d typically returns (xbins, ybins)
        # but image conventions are often (ybins, xbins) i.e. (rows, columns)
        psf_image = be.transpose(psf_image)

        if self.normalize:
            psf_sum = be.sum(psf_image)
            if psf_sum > 0:
                psf_image = psf_image / psf_sum
            # If sum is zero (e.g. no rays hit), it remains an array of zeros.

        # Ensure PSF is centered for visualization if BasePSF assumes it
        # The histogram is based on data range, so it should be okay.
        # BasePSF visualization finds bounds and zooms, so exact centering here
        # might not be strictly necessary if the data itself is centered.

        return psf_image, x_edges, y_edges

    def _get_psf_units(self, image_data_shape):
        """Calculates the physical extent (units) of the PSF image for plotting.

        This method is called by `BasePSF.view()` to determine axis labels.
        It uses the bin edges from the histogram computation.

        Args:
            image_data_shape (tuple): The shape of the PSF image data
                (often a zoomed/cropped version from `BasePSF.view`). This
                argument is present to match the signature in `FFTPSF`, but
                for `GeometricPSF`, the extent is directly derived from
                `self.x_edges` and `self.y_edges` corresponding to the
                *original, unzoomed* PSF data. The `BasePSF.view` method
                handles the zooming and passes the zoomed image shape.
                Here, we return the extent of the *original* data from which
                the `image_data_shape` was derived.

        Returns:
            tuple[float, float]: A tuple containing the physical
            total width and total height of the *original* PSF image area,
            in micrometers (assuming spot diagram coordinates are in µm).
        """
        # The spot diagram coordinates are typically in micrometers.
        # x_edges and y_edges store the boundaries of the bins.
        # The total extent is the difference between the last and first edge.

        # BasePSF.view calls this with psf_zoomed.shape.
        # We need to provide the extent that corresponds to that psf_zoomed.

        # Physical width of one original bin:
        dx_bin = (self.x_edges[1] - self.x_edges[0]) if len(self.x_edges) > 1 else 0
        dy_bin = (self.y_edges[1] - self.y_edges[0]) if len(self.y_edges) > 1 else 0

        # extent of the passed image_data (which is psf_zoomed from BasePSF)
        x_extent = image_data_shape[1] * be.to_numpy(dx_bin)
        y_extent = image_data_shape[0] * be.to_numpy(dy_bin)

        return x_extent, y_extent

    def strehl_ratio(self):
        """Computes the Strehl ratio of the PSF.

        For a geometric PSF, the concept of Strehl ratio (comparison to
        diffraction-limited peak) is not directly applicable in the same way
        as for wave optics PSF. This method could return a measure of peak
        concentration, or simply acknowledge it's not standard.

        Currently, this returns the peak value of the computed PSF.
        If self.normalize was True (default), this is peak of sum-normalized PSF.
        If self.normalize was False, this is peak of raw counts.

        Returns:
            float: The peak value of the computed PSF.
        """
        if self.psf is None:
            raise RuntimeError("PSF has not been computed.")
        return be.max(self.psf)

    # Note: The _get_working_FNO method is inherited from BasePSF.
    # The view method is inherited from BasePSF.
