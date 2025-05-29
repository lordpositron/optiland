import pytest
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap as get_matplotlib_cmap
from unittest.mock import MagicMock, patch

from optiland.optic import Optic
from optiland.surfaces import Surface
from optiland.geometries import StandardGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.materials.abbe import AbbeMaterial
from optiland.materials.material import Material  # For mocking

# Import your coloring module components
from optiland.visualization.coloring import (
    get_colormap,
    get_coloring_strategy,
    DEFAULT_LENS_COLOR,
    ColoringStrategy,  # Base class for isinstance checks
    DefaultColoring,
    AbbeNumberColoring,
    RefractiveIndexColoring,
    MaterialNameColoring,
    CurvatureSignColoring,
    SurfaceRoleColoring,
    COLORING_STRATEGIES,
)

# --- Fixtures ---


@pytest.fixture
def basic_optic():
    optic = Optic(name="TestOptic")
    optic.add_wavelength(
        value=0.55, is_primary=True
    )  # Primary wavelength for index calc
    return optic


@pytest.fixture
def coordinate_system_mock():
    # Using default CoordinateSystem for tests where its properties don't matter.
    # Import here to keep it local to where it's needed if not used globally.
    from optiland.coordinate_system import CoordinateSystem

    return CoordinateSystem()


@pytest.fixture
def surface_mock(coordinate_system_mock):
    # Basic surface, customize material and geometry in tests
    geometry = StandardGeometry(coordinate_system=coordinate_system_mock, radius=50.0)
    material = IdealMaterial(1.5)  # Positional argument for index
    # material_pre is not used by current strategies but good for completeness
    return Surface(
        geometry=geometry, material_pre=IdealMaterial(1.0), material_post=material
    )


# --- Tests for Helper Functions ---


def test_get_colormap_valid():
    cmap = get_colormap("viridis")
    assert isinstance(cmap, mcolors.Colormap)


def test_get_colormap_invalid_fallback():
    cmap = get_colormap("non_existent_map")
    assert isinstance(cmap, mcolors.Colormap)  # Should fallback to viridis
    # Check if it's actually viridis (or some default)
    # This comparison is a bit tricky, could compare names if available or check a known color
    assert cmap.name == "viridis"


def test_get_coloring_strategy_valid():
    for name, strategy_class in COLORING_STRATEGIES.items():
        strategy_instance = get_coloring_strategy(name)
        assert isinstance(strategy_instance, strategy_class)


def test_get_coloring_strategy_invalid_fallback():
    strategy_instance = get_coloring_strategy("non_existent_strategy")
    assert isinstance(strategy_instance, DefaultColoring)


def test_get_coloring_strategy_default():
    strategy_instance = get_coloring_strategy()  # No arg
    assert isinstance(strategy_instance, DefaultColoring)


# --- Tests for Coloring Strategies ---


class TestDefaultColoring:
    def test_returns_default_color(self, surface_mock, basic_optic):
        strategy = DefaultColoring()
        color = strategy.get_color([surface_mock], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_returns_default_color_empty_surfaces(self, basic_optic):
        strategy = DefaultColoring()
        color = strategy.get_color([], basic_optic)
        assert color == DEFAULT_LENS_COLOR


class TestAbbeNumberColoring:
    def test_abbe_material_valid(self, basic_optic, coordinate_system_mock):
        strategy = AbbeNumberColoring()
        abbe_mat = AbbeMaterial(index=1.5, abbe_number=60.0, name="TestAbbeGlass")
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=abbe_mat,
        )

        color = strategy.get_color([surf], basic_optic, colormap_name="viridis")
        assert isinstance(color, tuple)
        assert len(color) == 4

        # Check normalization (60 is between 20 and 85)
        norm = mcolors.Normalize(vmin=20, vmax=85)
        expected_color = get_matplotlib_cmap("viridis")(norm(60.0))
        assert color == expected_color

    def test_material_with_abbe_attribute(self, basic_optic, coordinate_system_mock):
        strategy = AbbeNumberColoring()
        # Mock a generic Material that happens to have an 'abbe' attribute
        mock_material = MagicMock(spec=Material)
        mock_material.name = "MockMaterialWithAbbe"  # Set attributes directly
        mock_material.abbe = 40.0

        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material,
        )
        color = strategy.get_color([surf], basic_optic, colormap_name="plasma")
        assert isinstance(color, tuple)
        norm = mcolors.Normalize(vmin=20, vmax=85)
        expected_color = get_matplotlib_cmap("plasma")(norm(40.0))
        assert color == expected_color

    def test_material_with_material_data(self, basic_optic, coordinate_system_mock):
        strategy = AbbeNumberColoring()
        mock_material = MagicMock(spec=Material)
        mock_material.name = "MockMaterialWithData"
        mock_material.material_data = {"Abbe number": 30.0}  # Simulate loaded data

        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material,
        )
        color = strategy.get_color([surf], basic_optic, colormap_name="cividis")
        assert isinstance(color, tuple)
        norm = mcolors.Normalize(vmin=20, vmax=85)
        expected_color = get_matplotlib_cmap("cividis")(norm(30.0))
        assert color == expected_color

    def test_no_abbe_info_fallback(self, basic_optic, coordinate_system_mock):
        strategy = AbbeNumberColoring()
        ideal_mat = IdealMaterial(1.6)  # No Abbe info
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=ideal_mat,
        )
        color = strategy.get_color([surf], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_empty_surfaces_fallback(self, basic_optic):
        strategy = AbbeNumberColoring()
        color = strategy.get_color([], basic_optic)
        assert color == DEFAULT_LENS_COLOR


class TestRefractiveIndexColoring:
    def test_ideal_material_valid(self, basic_optic, coordinate_system_mock):
        strategy = RefractiveIndexColoring()
        # primary wavelength is 0.55 um
        ideal_mat = IdealMaterial(1.7)
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=ideal_mat,
        )

        color = strategy.get_color([surf], basic_optic, colormap_name="magma")
        assert isinstance(color, tuple)
        assert len(color) == 4
        norm = mcolors.Normalize(vmin=1.4, vmax=2.0)
        expected_color = get_matplotlib_cmap("magma")(norm(1.7))
        assert color == expected_color

    def test_abbe_material_n_method(self, basic_optic, coordinate_system_mock):
        strategy = RefractiveIndexColoring()
        abbe_mat = AbbeMaterial(index=1.6, abbe_number=60)  # .n() should return 1.6
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=abbe_mat,
        )

        color = strategy.get_color([surf], basic_optic, colormap_name="inferno")
        assert isinstance(color, tuple)
        assert len(color) == 4
        norm = mcolors.Normalize(vmin=1.4, vmax=2.0)
        expected_color = get_matplotlib_cmap("inferno")(norm(1.6))
        assert color == expected_color

    def test_material_n_method_exception_fallback(
        self, basic_optic, coordinate_system_mock
    ):
        strategy = RefractiveIndexColoring()
        # Need to ensure MagicMock for Material can be instantiated if Material itself requires 'name'
        # However, spec=Material should handle this if methods are just called.
        # If Material's __init__ is complex, might need `autospec=True` or mock Material's constructor.
        # For now, assuming .n() is the primary interaction.
        mock_material = MagicMock(spec=Material)
        mock_material.n.side_effect = Exception("Failed to get n")
        # If Material(name="foo") is required for mock_material to be valid based on spec:
        # mock_material = MagicMock(spec=Material(name="TestMaterialForMock"))
        # mock_material.n.side_effect = Exception("Failed to get n")

        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material,
        )
        color = strategy.get_color([surf], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_empty_surfaces_fallback(self, basic_optic):
        strategy = RefractiveIndexColoring()
        color = strategy.get_color([], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_no_optic_fallback(self, surface_mock):
        strategy = RefractiveIndexColoring()
        color = strategy.get_color([surface_mock], None)  # No optic
        assert color == DEFAULT_LENS_COLOR


class TestMaterialNameColoring:
    def test_known_material_name(self, basic_optic, coordinate_system_mock):
        strategy = MaterialNameColoring()
        # N-BK7 is in the default map
        mock_material = MagicMock(spec=Material)
        mock_material.name = "N-BK7"
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material,
        )

        color = strategy.get_color([surf], basic_optic)
        assert color == strategy.material_color_map["N-BK7"]

    def test_unknown_material_with_colormap(self, basic_optic, coordinate_system_mock):
        strategy = MaterialNameColoring()
        mock_material = MagicMock(spec=Material)
        mock_material.name = "MyCustomGlass"
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material,
        )

        # Test with a colormap
        color1 = strategy.get_color([surf], basic_optic, colormap_name="viridis")
        assert isinstance(color1, tuple)
        assert len(color1) == 4
        assert color1 != DEFAULT_LENS_COLOR  # Should be from colormap

        # Test if color is cached and consistent
        color2 = strategy.get_color([surf], basic_optic, colormap_name="viridis")
        assert color1 == color2

        # Test a different unknown material gets a different color (likely, hash collision is rare)
        mock_material2 = MagicMock(spec=Material)
        mock_material2.name = "AnotherCustomGlass"
        surf2 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material2,
        )
        color3 = strategy.get_color([surf2], basic_optic, colormap_name="viridis")
        assert color3 != color1

    def test_unknown_material_no_colormap_fallback(
        self, basic_optic, coordinate_system_mock
    ):
        strategy = MaterialNameColoring()
        mock_material = MagicMock(spec=Material)
        mock_material.name = "UnknownGlass"
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=mock_material,
        )

        color = strategy.get_color(
            [surf], basic_optic, colormap_name=None
        )  # No colormap
        assert color == DEFAULT_LENS_COLOR

    def test_ideal_air_material(self, basic_optic, coordinate_system_mock):
        strategy = MaterialNameColoring()
        air_mat = IdealMaterial(1.0)  # Should be identified as "AIR"
        surf = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=air_mat,
        )
        color = strategy.get_color([surf], basic_optic)
        assert color == strategy.material_color_map["AIR"]

    def test_empty_surfaces_fallback(self, basic_optic):
        strategy = MaterialNameColoring()
        color = strategy.get_color([], basic_optic)
        assert color == DEFAULT_LENS_COLOR


class TestCurvatureSignColoring:
    # Helper to create a surface with a given radius
    def _make_surf(self, radius, cs_mock):  # Added cs_mock
        geom = StandardGeometry(coordinate_system=cs_mock, radius=radius)
        return Surface(
            geometry=geom,
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )

    @pytest.mark.parametrize(
        "r1, r2, category_idx",
        [
            (10, -10, 0),  # Biconvex
            (-10, 10, 1),  # Biconcave
            (0, -10, 2),  # Plano-convex (flat first)
            (10, 0, 2),  # Plano-convex (flat second)
            (0, 10, 3),  # Plano-concave (flat first)
            (-10, 0, 3),  # Plano-concave (flat second)
            # Corrected expected categories based on code: abs(s1) < abs(s2) is Positive (4)
            (
                20,
                10,
                5,
            ),  # s1=20, s2=10. abs(20) < abs(10) is False. Code assigns category 5.
            (
                10,
                20,
                4,
            ),  # s1=10, s2=20. abs(10) < abs(20) is True. Code assigns category 4.
            (
                -10,
                -20,
                4,
            ),  # Positive Meniscus (R1 < R2, both negative, abs(R1)<abs(R2) for code's positive)
            (
                -20,
                -10,
                5,
            ),  # Negative Meniscus (R1 > R2, both negative, abs(R1)>abs(R2) for code's negative)
        ],
    )
    def test_lens_types(
        self, r1, r2, category_idx, basic_optic, coordinate_system_mock
    ):  # Added cs_mock
        strategy = CurvatureSignColoring()
        s1 = self._make_surf(r1, coordinate_system_mock)
        s2 = self._make_surf(r2, coordinate_system_mock)

        # Use a known discrete colormap
        cmap = get_matplotlib_cmap("tab10")
        num_categories = 6  # As defined in strategy
        norm_val = category_idx / (num_categories - 1) if num_categories > 1 else 0.5
        expected_color = cmap(norm_val)

        color = strategy.get_color([s1, s2], basic_optic, colormap_name="tab10")
        assert color == expected_color

    def test_invalid_surface_count(self, basic_optic, coordinate_system_mock):
        strategy = CurvatureSignColoring()
        s1 = self._make_surf(10, coordinate_system_mock)
        assert strategy.get_color([s1], basic_optic) == DEFAULT_LENS_COLOR  # Too few
        s3 = self._make_surf(-10, coordinate_system_mock)
        assert (
            strategy.get_color([s1, s1, s3], basic_optic) == DEFAULT_LENS_COLOR
        )  # Too many
        assert strategy.get_color([], basic_optic) == DEFAULT_LENS_COLOR

    def test_non_numeric_radii_fallback(self, basic_optic, coordinate_system_mock):
        strategy = CurvatureSignColoring()
        s1 = self._make_surf(10, coordinate_system_mock)
        s2 = self._make_surf(0, coordinate_system_mock)  # Initially valid
        s2.geometry.radius = "invalid"  # Now invalid

        color = strategy.get_color([s1, s2], basic_optic)
        assert color == DEFAULT_LENS_COLOR

        s1_invalid = self._make_surf(0, coordinate_system_mock)
        s1_invalid.geometry.radius = "invalid_too"
        s2_valid = self._make_surf(10, coordinate_system_mock)
        color2 = strategy.get_color([s1_invalid, s2_valid], basic_optic)
        assert color2 == DEFAULT_LENS_COLOR


class TestSurfaceRoleColoring:
    def test_first_surface_is_stop(self, basic_optic, coordinate_system_mock):
        strategy = SurfaceRoleColoring()
        s1 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )
        s1.is_stop = True
        s2 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )

        color = strategy.get_color([s1, s2], basic_optic)
        assert color == (0.9, 0.2, 0.2, 0.6)  # Reddish

    def test_last_surface_is_stop(self, basic_optic, coordinate_system_mock):
        strategy = SurfaceRoleColoring()
        s1 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )
        s2 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )
        s2.is_stop = True

        color = strategy.get_color([s1, s2], basic_optic)
        assert color == (0.2, 0.2, 0.9, 0.6)  # Bluish

    def test_no_stop_surface_in_list(self, basic_optic, coordinate_system_mock):
        strategy = SurfaceRoleColoring()
        s1 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )
        s2 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )

        color = strategy.get_color([s1, s2], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_empty_surfaces_fallback(self, basic_optic):
        strategy = SurfaceRoleColoring()
        color = strategy.get_color([], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_single_surface_not_stop(self, basic_optic, coordinate_system_mock):
        strategy = SurfaceRoleColoring()
        s1 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )
        s1.is_stop = False
        color = strategy.get_color([s1], basic_optic)
        assert color == DEFAULT_LENS_COLOR

    def test_single_surface_is_stop(self, basic_optic, coordinate_system_mock):
        strategy = SurfaceRoleColoring()
        s1 = Surface(
            geometry=StandardGeometry(coordinate_system_mock, radius=0.0),
            material_pre=IdealMaterial(1.0),
            material_post=IdealMaterial(1.5),
        )
        s1.is_stop = True
        color = strategy.get_color([s1], basic_optic)  # First surface is stop
        assert color == (0.9, 0.2, 0.2, 0.6)
