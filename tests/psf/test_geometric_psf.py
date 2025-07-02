"""Unit tests for the GeometricPSF class.

This module contains tests for the geometric point spread function
functionality in optiland.psf.geometric.

Kramer Harrison, 2025
"""

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.psf.geometric import GeometricPSF
from optiland.samples.simple import SingletStopSurf2
from optiland.surfaces import ObjectSurface, Surface
from optiland.materials import IdealMaterial
from optiland.geometries import Plane


@pytest.fixture
def simple_optic():
    """Provides a simple Optic instance for testing."""
    # Object surface at infinity, material is Air (n=1)
    obj_geometry = Plane(cs_transform=(0,0,be.inf)) # z=inf for object at infinity
    air = IdealMaterial(index=1.0)
    obj = ObjectSurface(geometry=obj_geometry, material_post=air)

    s1 = Surface(radius=50, thickness=5, material_name="N-BK7")
    s2 = Surface(radius=-50, thickness=100)
    return Optic(surfaces=[obj, s1, s2])


@pytest.fixture
def default_geometric_psf_instance(simple_optic):
    """Provides a default GeometricPSF instance."""
    return GeometricPSF(
        optic=simple_optic,
        field=(0, 0.1),  # Slight off-axis field
        wavelength=0.55,
        num_rays=100, # Keep low for faster tests
        bins=32,      # Keep low for faster tests
    )


def test_geometric_psf_instantiation(default_geometric_psf_instance):
    """Tests basic instantiation of GeometricPSF."""
    psf_instance = default_geometric_psf_instance
    assert psf_instance is not None
    assert psf_instance.optic is not None
    assert psf_instance.fields == [(0, 0.1)]
    assert psf_instance.wavelengths == [0.55]
    assert psf_instance.num_rays == 100
    assert psf_instance.bins == 32


def test_geometric_psf_compute_psf_output_type_and_shape(default_geometric_psf_instance):
    """Tests the output type and shape of the computed PSF."""
    psf_instance = default_geometric_psf_instance
    assert psf_instance.psf is not None
    assert isinstance(psf_instance.psf, be.ndarray_type)
    assert psf_instance.psf.shape == (psf_instance.bins, psf_instance.bins)
    assert psf_instance.x_edges is not None
    assert psf_instance.y_edges is not None
    assert len(psf_instance.x_edges) == psf_instance.bins + 1
    assert len(psf_instance.y_edges) == psf_instance.bins + 1


def test_geometric_psf_normalization_sum_to_one(simple_optic):
    """Tests that the PSF is normalized to sum to 1 when normalize=True."""
    psf_instance = GeometricPSF(
        optic=simple_optic,
        field=(0, 0),
        wavelength=0.55,
        num_rays=100,
        bins=32,
        normalize=True,
    )
    psf_sum = be.sum(psf_instance.psf)
    # Using be.isclose for floating point comparison
    assert be.isclose(psf_sum, be.array(1.0)), f"PSF sum is {psf_sum}, expected 1.0"


def test_geometric_psf_normalization_false(simple_optic):
    """Tests PSF when normalize=False (sum of raw ray counts)."""
    # It's a bit hard to predict the exact sum of raw ray counts if some rays miss
    # or vignettes, but it should be <= num_rays.
    # If all rays land in bins, it should be num_rays.
    # For this test, we mainly check it's NOT close to 1.0 unless num_rays is 1.
    num_rays_test = 100
    psf_instance = GeometricPSF(
        optic=simple_optic,
        field=(0, 0), # On-axis, more rays should land
        wavelength=0.55,
        num_rays=num_rays_test,
        bins=32,
        normalize=False,
    )
    psf_sum = be.sum(psf_instance.psf)
    assert psf_sum > 1.0 # Should be significantly larger than 1
    # Max sum is num_rays_test. It can be less due to vignetting or rays missing bins.
    assert psf_sum <= num_rays_test


def test_geometric_psf_get_psf_units(default_geometric_psf_instance):
    """Tests the _get_psf_units method."""
    psf_instance = default_geometric_psf_instance
    # Simulate a "zoomed" image shape, though for GeometricPSF, it uses original bin sizes.
    # The shape passed is the shape of the data that will be displayed,
    # which is psf_zoomed in BasePSF.view().
    # Let's assume a case where it's cropped slightly.
    cropped_shape = (psf_instance.bins - 5, psf_instance.bins - 5)

    x_extent, y_extent = psf_instance._get_psf_units(cropped_shape)

    assert isinstance(x_extent, be.ndarray_type) or isinstance(x_extent, float)
    assert isinstance(y_extent, be.ndarray_type) or isinstance(y_extent, float)

    # Expected extent calculation:
    # dx_bin = (psf_instance.x_edges[1] - psf_instance.x_edges[0])
    # dy_bin = (psf_instance.y_edges[1] - psf_instance.y_edges[0])
    # expected_x_extent = cropped_shape[1] * dx_bin
    # expected_y_extent = cropped_shape[0] * dy_bin
    # Need to convert to numpy for comparison if they are backend arrays

    # Check positivity. Actual values depend on spot diagram extent.
    assert be.to_numpy(x_extent) > 0
    assert be.to_numpy(y_extent) > 0

    # Test with full shape (no cropping by view method)
    full_shape = psf_instance.psf.shape
    x_extent_full, y_extent_full = psf_instance._get_psf_units(full_shape)

    expected_x_extent_full = be.to_numpy(psf_instance.x_edges[-1] - psf_instance.x_edges[0])
    expected_y_extent_full = be.to_numpy(psf_instance.y_edges[-1] - psf_instance.y_edges[0])

    assert be.isclose(be.to_numpy(x_extent_full), expected_x_extent_full)
    assert be.isclose(be.to_numpy(y_extent_full), expected_y_extent_full)


def test_geometric_psf_strehl_ratio_normalized(simple_optic):
    """Tests Strehl ratio for a normalized PSF."""
    psf_instance = GeometricPSF(
        optic=simple_optic,
        field=(0, 0),
        wavelength=0.55,
        num_rays=200, # More rays for a better peak
        bins=64,
        normalize=True,
    )
    strehl = psf_instance.strehl_ratio()
    assert isinstance(strehl, (float, be.ndarray_type))
    # For a sum-normalized PSF, Strehl is the peak value. It should be <= 1.
    assert be.to_numpy(strehl) > 0
    assert be.to_numpy(strehl) <= 1.0
    # It should be equal to the max value in the PSF array
    assert be.isclose(strehl, be.max(psf_instance.psf))


def test_geometric_psf_strehl_ratio_unnormalized(simple_optic):
    """Tests Strehl ratio for an unnormalized PSF."""
    psf_instance = GeometricPSF(
        optic=simple_optic,
        field=(0, 0),
        wavelength=0.55,
        num_rays=200,
        bins=64,
        normalize=False,
    )
    strehl = psf_instance.strehl_ratio()
    assert isinstance(strehl, (float, be.ndarray_type))
    # For an unnormalized PSF, Strehl is the peak count.
    assert be.to_numpy(strehl) > 0
    # Max possible peak is num_rays, but usually less concentrated.
    assert be.to_numpy(strehl) <= psf_instance.num_rays
    assert be.isclose(strehl, be.max(psf_instance.psf))


def test_geometric_psf_with_singlet_lens_sample():
    """Tests GeometricPSF with a predefined lens sample."""
    optic = SingletStopSurf2() # Use the imported class
    psf_instance = GeometricPSF(
        optic=optic,
        field=(0, 0.5), # Off-axis field
        wavelength=0.656, # C-light (red)
        num_rays=150,
        bins=40,
    )
    assert psf_instance.psf is not None
    assert psf_instance.psf.shape == (40, 40)
    psf_sum = be.sum(psf_instance.psf)
    assert be.isclose(psf_sum, be.array(1.0)) # Default is normalize=True


def test_geometric_psf_no_rays_land():
    """Tests scenario where no rays might land in bins or ray tracing fails.
    This can be simulated by a very off-axis field for a simple lens.
    """
    obj_geometry = Plane(cs_transform=(0,0,be.inf))
    air = IdealMaterial(index=1.0)
    obj = ObjectSurface(geometry=obj_geometry, material_post=air)
    s1 = Surface(radius=50, thickness=5, material_name="N-BK7", aperture_radius=5) # Changed StandardSurface
    s2 = Surface(radius=-50, thickness=100, aperture_radius=5) # Changed StandardSurface
    optic = Optic(surfaces=[obj, s1, s2])

    # A field so far off-axis that few or no rays make it through
    # or they land very far from the central region covered by default binning range.
    psf_instance = GeometricPSF(
        optic=optic,
        field=(0, 30.0), # Very large field angle
        wavelength=0.55,
        num_rays=50,
        bins=16,
        normalize=True,
    )
    assert psf_instance.psf is not None
    assert psf_instance.psf.shape == (16, 16)
    psf_sum = be.sum(psf_instance.psf)

    # If no rays land, sum should be 0. If normalized, it might still be 0 or Nan
    # The implementation of GeometricPSF returns 0s if sum is 0 before normalization.
    # If normalize=True and psf_sum was 0, psf_image remains 0s.
    assert be.isclose(psf_sum, be.array(0.0)) or be.isnan(psf_sum)

    # Test _get_psf_units in this case
    # If x_edges/y_edges are default due to no rays, extents might be based on those.
    # Current GeometricPSF._compute_psf falls back to default edges if no rays.
    # These default edges are be.linspace(-1, 1, self.bins + 1)
    x_extent, y_extent = psf_instance._get_psf_units(psf_instance.psf.shape)

    # Based on default edges [-1, 1], total width/height is 2.
    # Pixel size = 2 / bins. Extent = shape * pixel_size.
    # If shape is (16,16) and bins=16, extent should be approx 2.
    expected_extent_val = 2.0
    assert be.isclose(be.to_numpy(x_extent), expected_extent_val)
    assert be.isclose(be.to_numpy(y_extent), expected_extent_val)

    strehl = psf_instance.strehl_ratio()
    assert be.isclose(be.to_numpy(strehl), be.array(0.0)) # Max of a zero array is 0


# More tests could be added for:
# - Different ray distributions (if they significantly alter PSF characteristics beyond spot diagram)
# - Edge cases for bins (e.g., bins=1, though SpotDiagram might have min requirements)
# - Interaction with optics having vignetting.
# - PSF centering if a more sophisticated centering logic were added.
# - Comparison with a known reference PSF if available (more of an integration/validation test).


# --- Tests for GeometricMTF integration ---
from optiland.mtf.geometric import GeometricMTF

@pytest.fixture
def default_geometric_mtf_instance(simple_optic):
    """Provides a default GeometricMTF instance using the new GeometricPSF."""
    return GeometricMTF(
        optic=simple_optic,
        fields=[(0, 0.1)], # List of fields
        wavelength=0.55,
        num_rays=100,    # For underlying GeometricPSF
        psf_bins=32,     # For underlying GeometricPSF
        num_points=64,   # For MTF curve
        max_freq=50,     # cycles/mm
    )

def test_geometric_mtf_instantiation_with_geometric_psf(default_geometric_mtf_instance):
    """Tests basic instantiation of GeometricMTF using GeometricPSF."""
    mtf_instance = default_geometric_mtf_instance
    assert mtf_instance is not None
    assert mtf_instance.optic is not None
    assert len(mtf_instance.resolved_fields) == 1
    assert mtf_instance.resolved_wavelength == 0.55
    assert hasattr(mtf_instance, 'psf_results')
    assert len(mtf_instance.psf_results) == 1
    assert isinstance(mtf_instance.psf_results[0], GeometricPSF)
    assert mtf_instance.psf_results[0].bins == 32
    assert mtf_instance.psf_results[0].num_rays == 100

def test_geometric_mtf_generates_mtf_data(default_geometric_mtf_instance):
    """Tests that GeometricMTF generates MTF data."""
    mtf_instance = default_geometric_mtf_instance
    assert mtf_instance.mtf is not None
    assert len(mtf_instance.mtf) == len(mtf_instance.resolved_fields)

    # Each field should have [tangential_mtf, sagittal_mtf]
    field_mtf_data = mtf_instance.mtf[0]
    assert isinstance(field_mtf_data, list)
    assert len(field_mtf_data) == 2

    tangential_mtf, sagittal_mtf = field_mtf_data
    assert isinstance(tangential_mtf, be.ndarray_type)
    assert isinstance(sagittal_mtf, be.ndarray_type)
    assert len(tangential_mtf) == mtf_instance.num_points
    assert len(sagittal_mtf) == mtf_instance.num_points

    # MTF values should be between 0 and 1 (inclusive)
    assert be.all(tangential_mtf >= 0) and be.all(tangential_mtf <= 1.0001) # Epsilon for float
    assert be.all(sagittal_mtf >= 0) and be.all(sagittal_mtf <= 1.0001) # Epsilon for float

    # MTF at zero frequency should ideally be 1.0
    assert be.isclose(tangential_mtf[0], be.array(1.0))
    assert be.isclose(sagittal_mtf[0], be.array(1.0))

def test_geometric_mtf_view_method_runs(default_geometric_mtf_instance, mocker):
    """Tests that the view() method of GeometricMTF runs without error (smoke test)."""
    mtf_instance = default_geometric_mtf_instance

    # Mock plt.show() to prevent actual plot windows during tests
    mocker.patch("matplotlib.pyplot.show")

    try:
        mtf_instance.view()
        mtf_instance.view(add_reference=True) # Test with reference
    except Exception as e:
        pytest.fail(f"GeometricMTF.view() raised an exception: {e}")

def test_geometric_mtf_with_multiple_fields(simple_optic):
    """Tests GeometricMTF with multiple field points."""
    fields_test = [(0,0), (0,0.1), (0.1,0.1)]
    mtf_instance = GeometricMTF(
        optic=simple_optic,
        fields=fields_test,
        wavelength=0.58,
        num_rays=80,
        psf_bins=24,
        num_points=32,
        max_freq="cutoff" # Test with cutoff frequency
    )
    assert len(mtf_instance.resolved_fields) == len(fields_test)
    assert len(mtf_instance.psf_results) == len(fields_test)
    assert len(mtf_instance.mtf) == len(fields_test)

    for i in range(len(fields_test)):
        assert isinstance(mtf_instance.psf_results[i], GeometricPSF)
        field_mtf_data = mtf_instance.mtf[i]
        assert len(field_mtf_data) == 2
        assert len(field_mtf_data[0]) == mtf_instance.num_points # Tangential
        assert len(field_mtf_data[1]) == mtf_instance.num_points # Sagittal
