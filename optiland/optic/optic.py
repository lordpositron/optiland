"""Optic Module

This is the core module of Optiland, which provides the class to define
generic optical systems. The Optic class encapsulates the core properties
of an optical system, such as the aperture, fields, surfaces, and
wavelengths. It also provides methods to draw the optical system, trace rays,
and perform paraxial and aberration analyses. Instances of the Optic class
are used as arguments to various analysis, optimization, and visualization
functions in Optiland.

Kramer Harrison, 2024
"""

from copy import deepcopy
from typing import Union

from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.apodization import BaseApodization
from optiland.fields.base import BaseFieldStrategy
from optiland.fields.field import Field
from optiland.fields.field_group import FieldGroup
from optiland.fields.solvers import ParaxialFieldSolver, RealFieldSolver
from optiland.fields.strategies import (
    AngleField,
    ImageSpaceField,
    ObjectHeightField,
)
from optiland.materials.base import BaseMaterial
from optiland.optic.optic_updater import OpticUpdater
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.rays import PolarizationState, RayGenerator
from optiland.raytrace.real_ray_tracer import RealRayTracer
from optiland.solves import SolveManager
from optiland.surfaces import ObjectSurface, SurfaceGroup
from optiland.visualization import LensInfoViewer, OpticViewer, OpticViewer3D
from optiland.wavelength import WavelengthGroup


class Optic:
    """The Optic class represents an optical system.

    Attributes:
        name (str, optional): An optional name for the optical system.
        aperture (Aperture | None): The aperture of the optical system.
        field_type (BaseFieldStrategy | None): The field strategy instance defining
            how field-dependent calculations are handled (e.g., based on object
            height or angle). Initially None, set via `set_field_type`.
        surface_group (SurfaceGroup): The group of surfaces in the optical
            system.
        fields (FieldGroup): The group of fields in the optical system.
        wavelengths (WavelengthGroup): The group of wavelengths in the optical
            system.
        paraxial (Paraxial): The paraxial analysis helper class for the
            optical system.
        aberrations (Aberrations): The aberrations analysis helper class for
            the optical system.
        ray_tracer (RealRayTracer): The real ray tracer instance.
        polarization (str | PolarizationState): The polarization state.
            Can be "ignore" or a PolarizationState object.
        apodization (BaseApodization | None): The apodization applied to the system.
        pickups (PickupManager): Manages pickup solves.
        solves (SolveManager): Manages optimization solves.
        obj_space_telecentric (bool): True if object space is telecentric.

    """

    def __init__(self, name: str = None):
        """Initialize an Optic instance.

        Args:
            name (str, optional): An optional name for the optical system.
                Defaults to None.

        """
        self.name = name
        self.reset()

    def _initialize_attributes(self):
        """Reset the optical system to its initial state."""
        self.aperture = None
        self.field_type: BaseFieldStrategy | None = (
            None  # Will hold a strategy instance
        )

        self.surface_group = SurfaceGroup()
        self.fields = FieldGroup()
        self.wavelengths = WavelengthGroup()

        self.paraxial = Paraxial(self)
        self.aberrations = Aberrations(self)
        self.ray_tracer = RealRayTracer(self)

        self.polarization = "ignore"

        self.apodization = None
        self.pickups = PickupManager(self)
        self.solves = SolveManager(self)
        self.obj_space_telecentric = False
        self._updater = OpticUpdater(self)

    def __add__(self, other):
        """Add two Optic objects together."""
        new_optic = deepcopy(self)
        new_optic.surface_group += other.surface_group
        return new_optic

    @property
    def primary_wavelength(self):
        """float: The primary wavelength in microns."""
        return self.wavelengths.primary_wavelength.value

    @property
    def object_surface(self):
        """Surface: The object surface instance."""
        for surface in self.surface_group.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self):
        """Surface: The image surface instance."""
        return self.surface_group.surfaces[-1]

    @property
    def total_track(self):
        """float: The total track length of the system."""
        return self.surface_group.total_track

    @property
    def polarization_state(self):
        """PolarizationState: The polarization state of the optic."""
        if self.polarization == "ignore":
            return None
        elif isinstance(self.polarization, PolarizationState):
            return self.polarization
        else:
            raise ValueError(
                "Invalid polarization state. Must be either "
                '"PolarizationState" or "ignore".',
            )

    def reset(self):
        """Reset the optical system to its initial state."""
        self._initialize_attributes()

    def add_surface(
        self,
        new_surface=None,
        surface_type="standard",
        comment="",
        index=None,
        is_stop=False,
        material="air",
        **kwargs,
    ):
        """Add a new surface to the optic.

        Args:
            new_surface (Surface, optional): The new surface to add. If not
                provided, a new surface will be created based on the other
                arguments.
            surface_type (str, optional): The type of surface to create.
            comment (str, optional): A comment for the surface.
            index (int, optional): The index at which to insert the new
                surface. If not provided, the surface will be appended to the
                end of the list.
            is_stop (bool, optional): Indicates if the surface is the aperture.
            material (str, optional): The material of the surface.
                Default is 'air'.
            **kwargs: Additional keyword arguments for surface-specific
                parameters such as radius, conic, dx, dy, rx, ry, aperture.

        Raises:
            ValueError: If index is not provided when defining a new surface.

        """
        self.surface_group.add_surface(
            new_surface=new_surface,
            surface_type=surface_type,
            comment=comment,
            index=index,
            is_stop=is_stop,
            material=material,
            **kwargs,
        )

    def add_field(self, y, x=0.0, vx=0.0, vy=0.0):
        """Add a field to the optical system.

        Args:
            y (float): The y-coordinate of the field.
            x (float, optional): The x-coordinate of the field.
                Defaults to 0.0.
            vx (float, optional): The x-component of the field's vignetting
                factor. Defaults to 0.0.
            vy (float, optional): The y-component of the field's vignetting
                factor. Defaults to 0.0.

        Raises:
            RuntimeError: If `Optic.field_type` has not been set via
                `set_field_type()` prior to calling this method.
            TypeError: If `Optic.field_type` is not a recognized strategy instance.

        """
        if not self.field_type:
            raise RuntimeError(
                "Optic.field_type must be set before adding fields. "
                "Call Optic.set_field_type() first."
            )

        field_type_str = ""
        if isinstance(self.field_type, ObjectHeightField):
            field_type_str = "object_height"
        elif isinstance(self.field_type, AngleField):
            field_type_str = "angle"
        else:
            # Should not happen if self.field_type is correctly initialized
            raise TypeError("Optic.field_type is not a recognized strategy instance.")

        new_field = Field(field_type_str, x, y, vx, vy)
        self.fields.add_field(new_field)

    def add_wavelength(self, value, is_primary=False, unit="um"):
        """Add a wavelength to the optical system.

        Args:
            value (float): The value of the wavelength.
            is_primary (bool, optional): Whether the wavelength is the primary
                wavelength. Defaults to False.
            unit (str, optional): The unit of the wavelength. Defaults to 'um'.

        """
        self.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type, value):
        """Set the aperture of the optical system.

        Args:
            aperture_type (str): The type of the aperture.
            value (float): The value of the aperture.

        """
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type: str):
        """Set the field definition strategy for the optical system.

        This method takes a string identifier ("object_height" or "angle") and
        instantiates the corresponding field strategy object. For direct
        object-space definitions like "object_height" or "angle", it creates
        `ObjectHeightField` or `AngleField` respectively. For image-space
        definitions like "paraxial_image_height" or "real_image_height", it
        composes an `ImageSpaceField` strategy with the appropriate solver
        (ParaxialFieldSolver or RealFieldSolver) and an underlying object-space
        strategy (AngleField for infinite conjugate, ObjectHeightField otherwise).

        The instantiated strategy is stored in `self.field_type` and used for
        all field-dependent calculations. The chosen strategy also validates
        the current optic state for compatibility.

        Args:
            field_type (str): The type of field strategy to use. Valid options are:
                - "object_height": Defines fields by object height.
                - "angle": Defines fields by angle (typically from entrance pupil).
                - "paraxial_image_height": Defines fields by paraxial image height.
                - "real_image_height": Defines fields by real image height.

        Raises:
            ValueError: If an invalid field_type string is provided, or if the
                        chosen strategy's `validate_optic_state` check fails,
                        or if an image-space strategy cannot determine the
                        underlying object-space strategy (e.g., object surface
                        not yet defined).
            RuntimeError: If components required by the chosen strategy are missing
                          (e.g. object surface for image space strategies).
        """
        strategy_instance: BaseFieldStrategy

        if field_type == "object_height":
            strategy_instance = ObjectHeightField()
        elif field_type == "angle":
            strategy_instance = AngleField()
        elif field_type in ("paraxial_image_height", "real_image_height"):
            # Determine solver
            if field_type == "paraxial_image_height":
                solver = ParaxialFieldSolver()
            else:  # real_image_height
                solver = RealFieldSolver()

            # Determine underlying base strategy
            if self.object_surface is None:
                raise RuntimeError(
                    f"Cannot set field type to '{field_type}' before an "
                    "object surface is defined. Needed for infinite conjugate check."
                )

            if self.object_surface.is_infinite:
                base_strategy = AngleField()
            else:
                base_strategy = ObjectHeightField()

            strategy_instance = ImageSpaceField(
                solver=solver, base_strategy=base_strategy
            )
        else:
            valid_types = (
                '"object_height", "angle", "paraxial_image_height", '
                'or "real_image_height"'
            )
            error_msg = (
                f"Invalid field_type string: '{field_type}'. "
                f"Must be one of {valid_types}."
            )
            raise ValueError(error_msg)

        # Validate optic state with the new strategy before assigning
        # This also validates the base_strategy for ImageSpaceField.
        strategy_instance.validate_optic_state(self)

        self.field_type = strategy_instance

    def set_radius(self, value, surface_number):
        """Set the radius of curvature of a surface.

        Args:
            value (float): The value of the radius.
            surface_number (int): The index of the surface.

        """
        self._updater.set_radius(value, surface_number)

    def set_conic(self, value, surface_number):
        """Set the conic constant of a surface.

        Args:
            value (float): The value of the conic constant.
            surface_number (int): The index of the surface.

        """
        self._updater.set_conic(value, surface_number)

    def set_thickness(self, value, surface_number):
        """Set the thickness of a surface.

        Args:
            value (float): The value of the thickness.
            surface_number (int): The index of the surface.

        """
        self._updater.set_thickness(value, surface_number)

    def set_index(self, value, surface_number):
        """Set the index of refraction of a surface.

        Args:
            value (float): The value of the index of refraction.
            surface_number (int): The index of the surface.

        """
        self._updater.set_index(value, surface_number)

    def set_material(self, material: BaseMaterial, surface_number: int):
        """Set the material of a surface.

        Args:
            material (BaseMaterial): The material.
            surface_number (int): The index of the surface.

        """
        self._updater.set_material(material, surface_number)

    def set_asphere_coeff(self, value, surface_number, aspher_coeff_idx):
        """Set the asphere coefficient on a surface.

        Args:
            value (float): The value of aspheric coefficient.
            surface_number (int): The index of the surface.
            aspher_coeff_idx (int): index of the aspheric coefficient on the
                surface.

        """
        self._updater.set_asphere_coeff(value, surface_number, aspher_coeff_idx)

    def set_polarization(self, polarization: Union[PolarizationState, str]):
        """Set the polarization state of the optic.

        Args:
            polarization (Union[PolarizationState, str]): The polarization
                state to set. It can be either a `PolarizationState` object or
                'ignore'.

        """
        self._updater.set_polarization(polarization)

    def set_apodization(self, apodization: BaseApodization):
        """Set the apodization of the optical system.

        Args:
            apodization (Apodization): The apodization object to set.

        """
        self._updater.set_apodization(apodization)

    def scale_system(self, scale_factor):
        """Scale the optical system by a given scale factor.

        Args:
            scale_factor (float): The factor by which to scale the system.

        """
        self._updater.scale_system(scale_factor)

    def update_paraxial(self):
        """Update the semi-aperture of the surfaces based on paraxial analysis."""
        self._updater.update_paraxial()

    def update_normalization(self, surface) -> None:
        """Update the normalization radius of non-spherical surfaces."""
        self._updater.update_normalization(surface)

    def update(self) -> None:
        """Update surface properties (pickups, solves, paraxial properties)."""
        self._updater.update()

    def image_solve(self):
        """Update image position for marginal ray to cross axis at image."""
        self._updater.image_solve()

    def flip(self):
        """Flip the optical system.

        This reverses the order of surfaces (excluding object and image planes),
        their geometries, and materials. Pickups and solves referencing surface
        indices are updated accordingly. The coordinate system is adjusted such
        that the new first optical surface (originally the last one in the
        flipped segment) is placed at z=0.0.
        """
        self._updater.flip()

    def draw(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=3,
        distribution="line_y",
        figsize=(10, 4),
        xlim=None,
        ylim=None,
        title=None,
        reference=None,
    ):
        """Draw a 2D representation of the optical system.

        Args:
            fields (str | list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str | list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 3.
            distribution (str, optional): The distribution of the rays.
                Defaults to 'line_y'.
            figsize (tuple, optional): The size of the figure. Defaults to
                (10, 4).
            xlim (tuple, optional): The x-axis limits of the plot. Defaults to
                None.
            ylim (tuple, optional): The y-axis limits of the plot. Defaults to
                None.
            title (str, optional): The title for the plot. Defaults to None.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        Returns:
            matplotlib.figure.Figure: The matplotlib Figure object.

        """
        viewer = OpticViewer(self)
        fig = viewer.view(
            fields,
            wavelengths,
            num_rays,
            distribution=distribution,
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
            title=title,
            reference=reference,
        )
        return fig

    def draw3D(
        self,
        fields="all",
        wavelengths="primary",
        num_rays=24,
        distribution="ring",
        figsize=(1200, 800),
        dark_mode=False,
        reference=None,
    ):
        """Draw a 3D representation of the optical system.

        Args:
            fields (str | list, optional): The fields to be displayed.
                Defaults to 'all'.
            wavelengths (str | list, optional): The wavelengths to be
                displayed. Defaults to 'primary'.
            num_rays (int, optional): The number of rays to be traced for each
                field and wavelength. Defaults to 2.
            distribution (str, optional): The distribution of the rays.
                Defaults to 'ring'.
            figsize (tuple, optional): The size of the figure. Defaults to
                (1200, 800).
            dark_mode (bool, optional): Whether to use dark mode. Defaults to
                False.
            reference (str, optional): The reference rays to plot. Options
                include "chief" and "marginal". Defaults to None.

        """
        viewer = OpticViewer3D(self)
        viewer.view(
            fields,
            wavelengths,
            num_rays,
            distribution=distribution,
            figsize=figsize,
            dark_mode=dark_mode,
            reference=reference,
        )

    def info(self):
        """Display the optical system information."""
        viewer = LensInfoViewer(self)
        viewer.view()

    def n(self, wavelength: Union[float, str] = "primary"):
        """Get refractive indices of materials for each space at a wavelength.

        Args:
            wavelength (float | str, optional): The wavelength in microns for
                which to calculate the refractive indices. Can be a float value
                or the string 'primary' to use the system's primary wavelength.
                Defaults to 'primary'.

        Returns:
            be.ndarray: An array of refractive indices for each space.

        """
        if wavelength == "primary":
            wavelength = self.primary_wavelength
        return self.surface_group.n(wavelength)

    def trace(self, Hx, Hy, wavelength, num_rays=100, distribution="hexapolar"):
        """Trace a distribution of rays through the optical system.

        Args:
            Hx (float | be.ndarray): The normalized x field coordinate(s).
            Hy (float | be.ndarray): The normalized y field coordinate(s).
            wavelength (float): The wavelength of the rays in microns.
            num_rays (int, optional): The number of rays to be traced.
                Defaults to 100.
            distribution (str | optiland.distribution.BaseDistribution, optional):
                The distribution of the rays. Can be a string identifier (e.g.,
                'hexapolar', 'uniform') or a Distribution object.
                Defaults to 'hexapolar'.

        Returns:
            RealRays: The RealRays object containing the traced rays.

        """
        return self.ray_tracer.trace(Hx, Hy, wavelength, num_rays, distribution)

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        """Trace generic rays through the optical system.

        Args:
            Hx (float | be.ndarray): The normalized x field coordinate(s).
            Hy (float | be.ndarray): The normalized y field coordinate(s).
            Px (float | be.ndarray): The normalized x pupil coordinate(s).
            Py (float | be.ndarray): The normalized y pupil coordinate(s).
            wavelength (float): The wavelength of the rays in microns.

        """
        return self.ray_tracer.trace_generic(Hx, Hy, Px, Py, wavelength)

    def to_dict(self):
        """Convert the optical system to a dictionary.

        Returns:
            dict: The dictionary representation of the optical system.

        """
        data = {
            "version": 1.0,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "fields": self.fields.to_dict(),
            "wavelengths": self.wavelengths.to_dict(),
            "apodization": self.apodization.to_dict() if self.apodization else None,
            "pickups": self.pickups.to_dict(),
            "solves": self.solves.to_dict(),
            "surface_group": self.surface_group.to_dict(),
        }

        data["wavelengths"]["polarization"] = self.polarization

        field_type_str = ""
        if isinstance(self.field_type, ObjectHeightField):
            field_type_str = "object_height"
        elif isinstance(self.field_type, AngleField):
            field_type_str = "angle"
        # else: self.field_type might be None if not set.
        # For serialization, it should ideally be set if fields are meaningful.

        data["fields"]["field_type"] = field_type_str
        data["fields"]["object_space_telecentric"] = self.obj_space_telecentric
        return data

    @classmethod
    def from_dict(cls, data):
        """Create an optical system from a dictionary.

        Args:
            data (dict): The dictionary representation of the optical system.

        Returns:
            Optic: The optical system.

        """
        optic = cls()
        optic.aperture = Aperture.from_dict(data["aperture"])
        optic.surface_group = SurfaceGroup.from_dict(data["surface_group"])
        optic.fields = FieldGroup.from_dict(data["fields"])
        optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])

        apodization_data = data.get("apodization")
        if apodization_data:
            optic.apodization = BaseApodization.from_dict(apodization_data)

        optic.pickups = PickupManager.from_dict(optic, data["pickups"])
        optic.solves = SolveManager.from_dict(optic, data["solves"])

        optic.polarization = data["wavelengths"]["polarization"]

        field_type_str_from_dict = data["fields"].get("field_type", None)
        if field_type_str_from_dict:  # Ensure it's not empty or None
            optic.set_field_type(field_type_str_from_dict)
        else:
            if optic.fields.fields:
                raise ValueError(
                    "Field type string missing in dictionary but fields are present."
                )

        optic.obj_space_telecentric = data["fields"]["object_space_telecentric"]

        optic.paraxial = Paraxial(optic)
        optic.aberrations = Aberrations(optic)
        optic.ray_generator = RayGenerator(optic)

        return optic
