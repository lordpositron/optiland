"""System Visualization Module

This module contains the OpticalSystem class for visualizing optical systems.

Kramer Harrison, 2024
"""

from optiland.visualization.lens import Lens2D, Lens3D
from optiland.visualization.mirror import Mirror3D
from optiland.visualization.surface import Surface2D, Surface3D
from optiland.visualization.coloring import get_coloring_strategy, DefaultColoring


class OpticalSystem:
    """A class to represent an optical system for visualization. The optical
    system contains surfaces and lenses.

    Args:
        optic (Optic): The `Optic` object representing the optical system.
        rays (Rays): An instance of `Rays2D` or `Rays3D` containing ray data.
        projection (str, optional): The type of projection, either "2d" or "3d".
            Defaults to "2d".
        coloring_scheme (str, optional): The name of the coloring strategy to
            use for lenses (e.g., "abbe_number", "material_name"). If None,
            the default coloring strategy is used. Defaults to None.
        colormap_name (str, optional): The name of the matplotlib colormap to
            be used by the chosen coloring strategy if it supports colormaps.
            Defaults to None.

    Attributes:
        optic (Optic): The optical system.
        rays (Rays): The ray data.
        projection (str): The projection type ("2d" or "3d").
        coloring_scheme (str): Name of the lens coloring scheme.
        colormap_name (str): Name of the colormap for the scheme.
        components (list): Stores identified optical components for plotting.
        component_registry (dict): Maps component types and projection to
            specific visualization classes (e.g., `Lens2D`, `Mirror3D`).

    Methods:
        plot(ax): Identifies and plots the components of the optical system.
    """

    def __init__(
        self, optic, rays, projection="2d", coloring_scheme=None, colormap_name=None
    ):
        self.optic = optic
        self.rays = rays
        self.projection = projection
        self.coloring_scheme = coloring_scheme
        self.colormap_name = colormap_name
        self.components = []  # initialize empty list of components

        if self.projection not in ["2d", "3d"]:
            raise ValueError("Invalid projection type. Must be '2d' or '3d'.")

        self.component_registry = {
            "lens": {"2d": Lens2D, "3d": Lens3D},
            "mirror": {"2d": Surface2D, "3d": Mirror3D},
            "surface": {"2d": Surface2D, "3d": Surface3D},
        }

    def plot(self, ax):
        """Identifies and plots the components of the optical system.

        The appearance of lenses can be customized via the `coloring_scheme`
        and `colormap_name` provided during initialization.

        Args:
            ax (matplotlib.axes.Axes or vtkRenderer): The matplotlib axis (for 2D)
                or VTK renderer (for 3D) on which to plot the components.
        """
        self._identify_components()
        for component in self.components:
            component.plot(ax)

    def _identify_components(self):
        """Identifies the components of the optical system and adds them to the
        list of components.
        """
        self.components = []
        n = self.optic.n()  # refractive indices
        num_surf = self.optic.surface_group.num_surfaces

        lens_surfaces = []

        for k, surf in enumerate(self.optic.surface_group.surfaces):
            # Get the surface extent
            extent = self.rays.r_extent[k]

            # Object surface
            if k == 0:
                if not surf.is_infinite:
                    self._add_component("surface", (surf, extent))

            # Image surface or paraxial surface
            elif k == num_surf - 1 or surf.surface_type == "paraxial":
                self._add_component("surface", (surf, extent))

            # Surface is a mirror
            elif surf.is_reflective:
                self._add_component("mirror", (surf, extent))

            # Front surface of a lens
            elif n[k] > 1:
                surface = self._get_lens_surface(surf, extent)
                lens_surfaces.append(surface)

            # Back surface of a lens
            elif n[k] == 1 and n[k - 1] > 1 and lens_surfaces:
                surface = self._get_lens_surface(surf, extent)
                lens_surfaces.append(surface)
                self._add_component("lens", lens_surfaces)

                lens_surfaces = []

        # add final lens, if any
        if lens_surfaces:
            self._add_component("lens", lens_surfaces)

    def _add_component(self, component_name, component_data):
        """Adds a component to the list of components for plotting.

        For "lens" components, it uses the instance's `coloring_scheme` and
        `colormap_name` to initialize the lens viewer with the appropriate
        coloring strategy.

        Args:
            component_name (str): The type of component (e.g., "lens", "surface", "mirror").
            component_data (any): Data required to initialize the component's
                visualization class. For "lens", this is a list of surface
                viewer objects (e.g., `Surface2D` instances). For "surface" or
                "mirror", this is typically a tuple `(actual_surface_object, extent)`.
        """
        if component_name in self.component_registry:
            component_class = self.component_registry[component_name][self.projection]
        else:
            raise ValueError(f"Component {component_name} not found in registry.")

        if component_name == "lens":
            strategy = get_coloring_strategy(self.coloring_scheme)
            self.components.append(
                component_class(
                    component_data, self.optic, strategy, self.colormap_name
                )
            )
        else:  # "surface" or "mirror"
            self.components.append(
                component_class(component_data[0], component_data[1])
            )

    def _get_lens_surface(self, surface, *args):
        """Gets the lens surface based on the projection type."""
        surface_class = self.component_registry["surface"][self.projection]
        return surface_class(surface, *args)
