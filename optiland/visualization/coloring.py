import abc
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np

from optiland.materials.material import Material
from optiland.materials.ideal import IdealMaterial
from optiland.materials.abbe import (
    AbbeMaterial,
)  # Assuming this exists for Abbe numbers

# Need Optic for type hint, but will cause circular import if imported directly
# from optiland.optic import Optic
# For type hinting, use a string literal 'Optic' or forward reference if needed.
# from optiland.surfaces import Surface # Similarly for Surface

DEFAULT_LENS_COLOR = (0.8, 0.8, 0.8, 0.6)  # Default grey color for lenses


def get_colormap(name="viridis"):
    """Retrieves a matplotlib colormap by name.

    Args:
        name (str, optional): The name of the colormap. Defaults to 'viridis'.

    Returns:
        matplotlib.colors.Colormap: The colormap object. Falls back to 'viridis'
            if the specified colormap is not found.
    """
    try:
        return mcm.get_cmap(name)
    except ValueError:
        print(f"Warning: Colormap '{name}' not found. Using default 'viridis'.")
        return mcm.get_cmap("viridis")


class ColoringStrategy(abc.ABC):
    """Abstract base class for defining coloring strategies for optical components.

    Subclasses must implement the `get_color` method.
    """

    @abc.abstractmethod
    def get_color(self, component_surfaces, optic, colormap_name=None):
        """Calculates the color for a given optical component.

        Args:
            component_surfaces (list[Surface]): A list of `Surface` objects
                that make up the component (e.g., a lens). For single-surface
                components like mirrors or standalone surfaces, this will be a
                list containing one `Surface`.
            optic (Optic): The parent `Optic` object, providing access to
                system-wide properties like wavelengths and materials.
            colormap_name (str, optional): The name of the matplotlib
                colormap to use (e.g., 'viridis', 'jet', 'tab10').
                If None, a default behavior (often a fixed color or a
                default colormap specific to the strategy) should be
                implemented by the subclass.

        Returns:
            tuple[float, float, float, float]: An RGBA color tuple, where each
            component is in the range [0, 1].
        """
        pass


class DefaultColoring(ColoringStrategy):
    """Default coloring strategy: uses a fixed standard gray color.

    This strategy ignores the `colormap_name` parameter.
    """

    def get_color(self, component_surfaces, optic, colormap_name=None):
        """Returns the default lens color.

        Args:
            component_surfaces (list[Surface]): A list of `Surface` objects.
                Unused in this strategy.
            optic (Optic): The parent `Optic` object. Unused in this strategy.
            colormap_name (str, optional): The name of the colormap. Unused.

        Returns:
            tuple[float, float, float, float]: The `DEFAULT_LENS_COLOR` RGBA tuple.
        """
        return DEFAULT_LENS_COLOR


class AbbeNumberColoring(ColoringStrategy):
    """Colors components based on the Abbe number of their material.

    This strategy assumes the component is a lens made of a single material,
    taken from the `material_post` attribute of the first surface in
    `component_surfaces`. The Abbe number is normalized to the range [20, 85]
    to map to the specified colormap. If no Abbe number can be determined,
    it falls back to `DEFAULT_LENS_COLOR`.
    """

    def get_color(self, component_surfaces, optic, colormap_name="viridis"):
        """Calculates color based on the Abbe number.

        Args:
            component_surfaces (list[Surface]): List of surfaces for the component.
                The material is taken from the first surface.
            optic (Optic): The parent `Optic` object. Unused by this strategy
                beyond being part of the signature.
            colormap_name (str, optional): Name of the colormap. Defaults to 'viridis'.

        Returns:
            tuple[float, float, float, float]: RGBA color from the colormap,
            or `DEFAULT_LENS_COLOR` on failure.
        """
        if not component_surfaces:
            return DEFAULT_LENS_COLOR

        material_obj = component_surfaces[0].material_post
        abbe_number = None

        if isinstance(material_obj, AbbeMaterial):
            abbe_number = material_obj.abbe
        elif isinstance(material_obj, Material):
            if (
                hasattr(material_obj, "material_data")
                and "Abbe number" in material_obj.material_data
            ):
                abbe_number = material_obj.material_data["Abbe number"]
            elif hasattr(material_obj, "abbe"):
                abbe_number = material_obj.abbe

        if abbe_number is None:
            return DEFAULT_LENS_COLOR

        colormap = get_colormap(colormap_name)
        # Normalize Abbe number. Typical range for optical glasses 20-85.
        norm = mcolors.Normalize(vmin=20, vmax=85, clip=True)
        return colormap(
            norm(float(abbe_number))
        )  # Ensure abbe_number is float for norm


class RefractiveIndexColoring(ColoringStrategy):
    """Colors components based on refractive index at the primary wavelength.

    Assumes the component is a lens of a single material, taken from the
    `material_post` of the first surface. The refractive index is evaluated
    at the `optic.primary_wavelength`. The index is normalized to the range
    [1.4, 2.0] to map to the colormap. Falls back to `DEFAULT_LENS_COLOR`
    if the index cannot be determined.
    """

    def get_color(self, component_surfaces, optic, colormap_name="viridis"):
        """Calculates color based on the refractive index.

        Args:
            component_surfaces (list[Surface]): List of surfaces for the component.
                Material is from the first surface.
            optic (Optic): The parent `Optic` object, used to get the
                `primary_wavelength`.
            colormap_name (str, optional): Name of the colormap. Defaults to 'viridis'.

        Returns:
            tuple[float, float, float, float]: RGBA color from the colormap,
            or `DEFAULT_LENS_COLOR` on failure.
        """
        if not component_surfaces or not optic:
            return DEFAULT_LENS_COLOR

        material_obj = component_surfaces[0].material_post
        primary_wavelength_um = optic.primary_wavelength

        refractive_index = None
        if isinstance(material_obj, (Material, IdealMaterial, AbbeMaterial)):
            try:
                refractive_index = material_obj.n(primary_wavelength_um)
            except Exception:
                pass

        if refractive_index is None:
            return DEFAULT_LENS_COLOR

        colormap = get_colormap(colormap_name)
        # Normalize refractive index. Typical range for optical glasses 1.4-2.0.
        norm = mcolors.Normalize(vmin=1.4, vmax=2.0, clip=True)
        return colormap(norm(float(refractive_index)))  # Ensure index is float


class MaterialNameColoring(ColoringStrategy):
    """Colors components based on their material name.

    It uses a predefined map for common material names (e.g., "N-BK7", "SF11").
    If a material name is not in this map and a `colormap_name` is provided,
    it generates a color from the colormap using a hash of the material name
    for consistency and caches it. This allows for distinct colors for
    previously unseen material names. If the name is unknown and no colormap
    is specified, it falls back to `DEFAULT_LENS_COLOR`.
    """

    def __init__(self):
        self.material_color_map = {
            "N-BK7": (0.5, 0.5, 0.9, 0.6),
            "SF11": (0.7, 0.4, 0.4, 0.6),
            "AIR": DEFAULT_LENS_COLOR,
            "MIRROR": DEFAULT_LENS_COLOR,
        }
        self.assigned_colors_cache = {}

    def get_color(self, component_surfaces, optic, colormap_name=None):
        """Calculates color based on the material name.

        Args:
            component_surfaces (list[Surface]): List of surfaces for the component.
                Material name is from the first surface.
            optic (Optic): The parent `Optic` object. Unused by this strategy.
            colormap_name (str, optional): Name of the colormap for unknown names.
                If None, unknown names get `DEFAULT_LENS_COLOR`.

        Returns:
            tuple[float, float, float, float]: RGBA color.
        """
        if not component_surfaces:
            return DEFAULT_LENS_COLOR

        material_obj = component_surfaces[0].material_post
        material_name = "Unknown"

        if hasattr(material_obj, "name"):
            material_name = material_obj.name
        elif isinstance(material_obj, IdealMaterial) and material_obj.index == 1.0:
            # Special handling for IdealMaterial that is air.
            material_name = "AIR"

        if material_name in self.material_color_map:
            return self.material_color_map[material_name]

        if colormap_name:
            if material_name not in self.assigned_colors_cache:
                colormap = get_colormap(colormap_name)
                if colormap.N > 20:  # Treat as continuous for this hashing purpose
                    color_val = (hash(material_name) % 256) / 255.0
                    self.assigned_colors_cache[material_name] = colormap(color_val)
                else:  # Discrete colormap
                    idx = hash(material_name) % colormap.N
                    self.assigned_colors_cache[material_name] = colormap(idx)
            return self.assigned_colors_cache[material_name]

        return DEFAULT_LENS_COLOR


class CurvatureSignColoring(ColoringStrategy):
    """Colors lenses based on the sign of curvature of their two surfaces.

    This strategy is intended for two-surface lenses. It categorizes lenses into:
    - Biconvex (Category 0)
    - Biconcave (Category 1)
    - Plano-convex (Category 2)
    - Plano-concave (Category 3)
    - Positive Meniscus (Category 4)
    - Negative Meniscus (Category 5)

    The category determines the color from the specified colormap (default 'tab10').
    Falls back to `DEFAULT_LENS_COLOR` if not a two-surface lens or if radii
    are invalid.
    """

    def get_color(self, component_surfaces, optic, colormap_name="tab10"):
        """Calculates color based on lens surface curvature signs.

        Args:
            component_surfaces (list[Surface]): List of two surfaces for the lens.
            optic (Optic): The parent `Optic` object. Unused by this strategy.
            colormap_name (str, optional): Name of the colormap. Defaults to 'tab10'.

        Returns:
            tuple[float, float, float, float]: RGBA color from the colormap,
            or `DEFAULT_LENS_COLOR` on failure/inapplicable.
        """
        if len(component_surfaces) != 2:
            return DEFAULT_LENS_COLOR

        s1_radius_val = component_surfaces[0].geometry.radius
        s2_radius_val = component_surfaces[1].geometry.radius

        try:
            s1_radius = float(s1_radius_val)
            s2_radius = float(s2_radius_val)
        except (TypeError, ValueError):
            return DEFAULT_LENS_COLOR

        if not (isinstance(s1_radius, (int, float)) and np.isfinite(s1_radius)) or not (
            isinstance(s2_radius, (int, float)) and np.isfinite(s2_radius)
        ):
            return DEFAULT_LENS_COLOR

        colormap = get_colormap(colormap_name)
        category = -1

        if s1_radius > 0 and s2_radius < 0:
            category = 0  # Biconvex
        elif s1_radius < 0 and s2_radius > 0:
            category = 1  # Biconcave
        elif s1_radius == 0 and s2_radius < 0:
            category = 2  # Plano-convex
        elif s1_radius > 0 and s2_radius == 0:
            category = 2  # Plano-convex
        elif s1_radius == 0 and s2_radius > 0:
            category = 3  # Plano-concave
        elif s1_radius < 0 and s2_radius == 0:
            category = 3  # Plano-concave
        elif (
            s1_radius > 0 and s2_radius > 0
        ):  # Positive/Negative Meniscus (both convex)
            category = 4 if abs(s1_radius) < abs(s2_radius) else 5
        elif (
            s1_radius < 0 and s2_radius < 0
        ):  # Positive/Negative Meniscus (both concave)
            category = 4 if abs(s1_radius) < abs(s2_radius) else 5

        if category != -1:
            num_categories = 6
            norm_val = category / (num_categories - 1) if num_categories > 1 else 0.5
            return colormap(norm_val)

        return DEFAULT_LENS_COLOR


class SurfaceRoleColoring(ColoringStrategy):
    """Colors components based on the role of their surfaces (e.g., stop surface).

    Simplified interpretation:
    - If the first surface of the component is the stop, colors it reddish.
    - If the last surface (of a multi-surface component) is the stop, colors it bluish.
    - Otherwise, returns `DEFAULT_LENS_COLOR`.
    This strategy currently does not use `colormap_name`.
    """

    def get_color(self, component_surfaces, optic, colormap_name=None):
        """Calculates color based on surface roles (e.g., stop).

        Args:
            component_surfaces (list[Surface]): List of surfaces.
            optic (Optic): The parent `Optic` object. Unused.
            colormap_name (str, optional): Name of the colormap. Unused.

        Returns:
            tuple[float, float, float, float]: Specific RGBA color if a role
            is identified, otherwise `DEFAULT_LENS_COLOR`.
        """
        if not component_surfaces:
            return DEFAULT_LENS_COLOR

        if component_surfaces[0].is_stop:
            return (0.9, 0.2, 0.2, 0.6)  # Reddish for stop lens (first surface)

        if len(component_surfaces) > 1 and component_surfaces[-1].is_stop:
            return (0.2, 0.2, 0.9, 0.6)  # Bluish for lens ending with stop

        return DEFAULT_LENS_COLOR


# Registry of available coloring strategies
COLORING_STRATEGIES = {
    "default": DefaultColoring,
    "abbe_number": AbbeNumberColoring,
    "refractive_index": RefractiveIndexColoring,
    "material_name": MaterialNameColoring,
    "curvature_sign": CurvatureSignColoring,
    "surface_role": SurfaceRoleColoring,
}


def get_coloring_strategy(name="default"):
    """Retrieves an instance of a specific coloring strategy by its registered name.

    Args:
        name (str, optional): The name of the coloring strategy. This name should
            be a key in the `COLORING_STRATEGIES` dictionary. Defaults to 'default'.

    Returns:
        ColoringStrategy: An instance of the requested coloring strategy.
            If the name is not found in the registry, a warning is printed, and an
            instance of `DefaultColoring` is returned.
    """
    strategy_class = COLORING_STRATEGIES.get(name.lower())
    if strategy_class:
        return strategy_class()
    else:
        print(f"Warning: Coloring strategy '{name}' not found. Using default strategy.")
        return DefaultColoring()
