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
from typing import Any, Optional, Union  # Dict removed

from optiland.aberrations import Aberrations
from optiland.aperture import Aperture
from optiland.apodization import BaseApodization
from optiland.fields import BaseFieldGroup, Field, make_field_group
from optiland.materials.base import BaseMaterial
from optiland.optic.optic_updater import OpticUpdater
from optiland.paraxial import Paraxial
from optiland.pickup import PickupManager
from optiland.rays import PolarizationState
from optiland.raytrace.real_ray_tracer import RealRayTracer
from optiland.solves import SolveManager
from optiland.surfaces import ObjectSurface, SurfaceGroup
from optiland.visualization import LensInfoViewer, OpticViewer, OpticViewer3D
from optiland.wavelength import WavelengthGroup


class Optic:
    """The Optic class represents an optical system.

    Attributes:
        name (Optional[str]): An optional name for the optical system.
        aperture (Optional[Aperture]): The aperture of the optical system.
        surface_group (SurfaceGroup): The group of surfaces in the optical
            system.
        fields (Optional[BaseFieldGroup]): The group of fields in the optical
            system. This will be an instance of a `BaseFieldGroup` subclass.
        wavelengths (WavelengthGroup): The group of wavelengths in the optical
            system.
        obj_space_telecentric (bool): Indicates if the system is telecentric in
            object space. This state is primarily managed by and synchronized
            with the `fields` object.
        paraxial (Paraxial): The paraxial analysis helper class for the
            optical system.
        aberrations (Aberrations): The aberrations analysis helper class for
            the optical system.
        ray_tracer (RealRayTracer): Handles real ray tracing for the system.
        polarization (Union[PolarizationState, str]): Polarization state for
            wavelengths, can be "ignore".
        apodization (Optional[BaseApodization]): Apodization function applied to
            the pupil.
        pickups (PickupManager): Manages pickups in the system.
        solves (SolveManager): Manages solves in the system.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        """Initializes a new Optic instance.

        Args:
            name (Optional[str], optional): An optional name for the optical
                system. Defaults to None.
        """
        self.name: Optional[str] = name
        self._initialize_attributes()

    def _initialize_attributes(self) -> None:
        """Resets the optical system to its initial state."""
        self.aperture: Optional[Aperture] = None
        self.surface_group: SurfaceGroup = SurfaceGroup()
        self.fields: Optional[BaseFieldGroup] = None
        self.wavelengths: WavelengthGroup = WavelengthGroup()

        self.paraxial: Paraxial = Paraxial(self)
        self.aberrations: Aberrations = Aberrations(self)
        self.ray_tracer: RealRayTracer = RealRayTracer(self)

        self.polarization: Union[PolarizationState, str] = "ignore"
        self.apodization: Optional[BaseApodization] = None
        self.pickups: PickupManager = PickupManager(self)
        self.solves: SolveManager = SolveManager(self)
        self.obj_space_telecentric: bool = False
        self._updater: OpticUpdater = OpticUpdater(self)

    def __add__(self, other: "Optic") -> "Optic":
        """Adds two Optic objects by concatenating their surface groups.

        Note: Other properties (fields, wavelengths, aperture) are taken from
        the first optic (`self`). This behavior might need refinement.

        Args:
            other (Optic): The other Optic object to add.

        Returns:
            Optic: A new Optic object with combined surfaces.
        """
        new_optic = deepcopy(self)
        new_optic.surface_group += other.surface_group
        return new_optic

    @property
    def primary_wavelength(self) -> float:
        """float: The primary wavelength in microns.

        Raises:
            ValueError: If no primary wavelength is set.
        """
        if self.wavelengths.primary_wavelength:
            return self.wavelengths.primary_wavelength.value
        raise ValueError("Primary wavelength not set.")

    @property
    def object_surface(self) -> Optional[ObjectSurface]:
        """Optional[ObjectSurface]: The object surface instance, if present."""
        for surface in self.surface_group.surfaces:
            if isinstance(surface, ObjectSurface):
                return surface
        return None

    @property
    def image_surface(self) -> Any:  # Surface type
        """Surface: The image surface instance (typically the last surface)."""
        if not self.surface_group.surfaces:
            return None
        return self.surface_group.surfaces[-1]

    @property
    def total_track(self) -> float:
        """float: Total track length from first to last surface."""
        return self.surface_group.total_track

    @property
    def polarization_state(self) -> Optional[PolarizationState]:
        """Optional[PolarizationState]: The polarization state. None if ignored."""
        if self.polarization == "ignore":
            return None
        elif isinstance(self.polarization, PolarizationState):
            return self.polarization
        raise ValueError(
            "Invalid polarization state. Must be a PolarizationState object "
            'or "ignore".'
        )

    def reset(self) -> None:
        """Resets the optical system to its default initial state."""
        self._initialize_attributes()

    def add_surface(
        self,
        new_surface: Optional[Any] = None,  # BaseSurface type
        surface_type: str = "standard",
        comment: str = "",
        index: Optional[int] = None,
        is_stop: bool = False,
        material: Union[str, BaseMaterial] = "air",
        **kwargs: Any,
    ) -> None:
        """Adds a new surface to the optic."""
        self.surface_group.add_surface(
            new_surface=new_surface,
            surface_type=surface_type,
            comment=comment,
            index=index,
            is_stop=is_stop,
            material=material,
            **kwargs,
        )

    def add_field(
        self, y: float, x: float = 0.0, vx: float = 0.0, vy: float = 0.0
    ) -> None:
        """Adds a field point to the optical system.

        A field type must be set using `set_field_type()` before adding fields.
        The x and y values are interpreted according to the active field type.

        Args:
            y (float): The y-coordinate or y-component of the field.
            x (float, optional): The x-coordinate or x-component of the field.
            vx (float, optional): Vignetting factor in x.
            vy (float, optional): Vignetting factor in y.

        Raises:
            ValueError: If no field type has been set.
        """
        if self.fields is None:
            raise ValueError(
                "A field type must be set using set_field_type() before adding fields."
            )
        new_field = Field(x=x, y=y, vignette_factor_x=vx, vignette_factor_y=vy)
        self.fields.add_field(new_field, self)

    def add_wavelength(
        self, value: float, is_primary: bool = False, unit: str = "um"
    ) -> None:
        """Adds a wavelength to the optical system."""
        self.wavelengths.add_wavelength(value, is_primary, unit)

    def set_aperture(self, aperture_type: str, value: float) -> None:
        """Sets the aperture of the optical system."""
        self.aperture = Aperture(aperture_type, value)

    def set_field_type(self, field_type_str: str) -> None:
        """Sets the type of field group for the optical system.

        This replaces any existing field group. Optic's telecentric flag
        is updated based on the new field group's default.

        Args:
            field_type_str (str): Identifier for the field type
                (e.g., "angle", "object_height").

        Raises:
            ValueError: If `field_type_str` is not recognized.
        """
        self.fields = make_field_group(field_type_str)
        if self.fields is None:
            raise ValueError(f"Unknown field type string: {field_type_str}")

        self.obj_space_telecentric = self.fields.telecentric
        try:
            self.fields.validate(self)
        except ValueError as e:
            # If validation fails (e.g. AngleFieldGroup with telecentric Optic)
            self.fields = None  # Reset fields
            raise e

    def set_obj_space_telecentric(self, is_telecentric: bool) -> None:
        """Sets object space telecentricity.

        Updates `self.obj_space_telecentric` and propagates to `self.fields`.
        The field group validates if it supports the state.

        Args:
            is_telecentric (bool): True if object space is telecentric.

        Raises:
            ValueError: If `self.fields` is not set, or if the field group
                        rejects the telecentric state.
        """
        if self.fields is None:
            raise ValueError(
                "Cannot set telecentricity: field type not yet defined. "
                "Call set_field_type() first."
            )

        self.obj_space_telecentric = is_telecentric
        try:
            self.fields.set_telecentric(is_telecentric)
        except ValueError as e:
            self.obj_space_telecentric = not is_telecentric  # Revert
            raise e

    def set_radius(self, value: float, surface_number: int) -> None:
        self._updater.set_radius(value, surface_number)

    def set_conic(self, value: float, surface_number: int) -> None:
        self._updater.set_conic(value, surface_number)

    def set_thickness(self, value: float, surface_number: int) -> None:
        self._updater.set_thickness(value, surface_number)

    def set_index(self, value: float, surface_number: int) -> None:
        self._updater.set_index(value, surface_number)

    def set_material(self, material: BaseMaterial, surface_number: int) -> None:
        self._updater.set_material(material, surface_number)

    def set_asphere_coeff(
        self, value: float, surface_number: int, asphere_coeff_idx: int
    ) -> None:
        self._updater.set_asphere_coeff(value, surface_number, asphere_coeff_idx)

    def set_polarization(self, polarization: Union[PolarizationState, str]) -> None:
        self._updater.set_polarization(polarization)

    def set_apodization(self, apodization: BaseApodization) -> None:
        self._updater.set_apodization(apodization)

    def scale_system(self, scale_factor: float) -> None:
        self._updater.scale_system(scale_factor)

    def update_paraxial(self) -> None:
        self._updater.update_paraxial()

    def update_normalization(self, surface: Any) -> None:  # Surface type
        self._updater.update_normalization(surface)

    def update(self) -> None:
        self._updater.update()

    def image_solve(self) -> None:
        self._updater.image_solve()

    def flip(self) -> None:
        self._updater.flip()

    def draw(
        self,
        fields: Union[str, list] = "all",
        wavelengths: Union[str, list] = "primary",
        num_rays: int = 3,
        distribution: str = "line_y",
        figsize: tuple = (10, 4),
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        title: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> Any:  # Figure type
        """Draws a 2D representation of the optical system with rays."""
        viewer = OpticViewer(self)
        fig = viewer.view(
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
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
        fields: Union[str, list] = "all",
        wavelengths: Union[str, list] = "primary",
        num_rays: int = 24,
        distribution: str = "ring",
        figsize: tuple = (1200, 800),
        dark_mode: bool = False,
        reference: Optional[str] = None,
    ) -> None:
        """Draws a 3D representation of the optical system with rays."""
        viewer = OpticViewer3D(self)
        viewer.view(
            fields=fields,
            wavelengths=wavelengths,
            num_rays=num_rays,
            distribution=distribution,
            figsize=figsize,
            dark_mode=dark_mode,
            reference=reference,
        )

    def info(self) -> None:
        """Displays a summary table of the optical system's properties."""
        viewer = LensInfoViewer(self)
        viewer.view()

    def n(self, wavelength: Union[float, str] = "primary") -> Any:  # be.ndarray
        """Gets refractive indices for each space at a wavelength."""
        wl_val = (
            self.primary_wavelength if wavelength == "primary" else float(wavelength)
        )
        return self.surface_group.n(wl_val)

    def trace(
        self,
        Hx: Any,  # float or be.ndarray
        Hy: Any,  # float or be.ndarray
        wavelength: float,
        num_rays: int = 100,
        distribution: str = "hexapolar",
    ) -> Any:  # RealRays
        """Traces a distribution of rays through the system."""
        return self.ray_tracer.trace(
            Hx=Hx,
            Hy=Hy,
            wavelength=wavelength,
            num_rays=num_rays,
            distribution=distribution,
        )

    def trace_generic(
        self,
        Hx: Any,  # float or be.ndarray
        Hy: Any,  # float or be.ndarray
        Px: Any,  # float or be.ndarray
        Py: Any,  # float or be.ndarray
        wavelength: float,
    ) -> Any:  # RealRays
        """Traces specific rays defined by field and pupil coordinates."""
        return self.ray_tracer.trace_generic(
            Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wavelength
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the optical system to a dictionary."""
        fields_dict = self.fields.to_dict() if self.fields else None

        data = {
            "version": 1.1,
            "name": self.name,
            "aperture": self.aperture.to_dict() if self.aperture else None,
            "fields": fields_dict,
            "wavelengths": self.wavelengths.to_dict(),
            "apodization": self.apodization.to_dict() if self.apodization else None,
            "pickups": self.pickups.to_dict(),
            "solves": self.solves.to_dict(),
            "surface_group": self.surface_group.to_dict(),
            "object_space_telecentric": self.obj_space_telecentric,
            "polarization": self.polarization,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Optic":
        """Creates an Optic instance from a dictionary representation."""
        optic = cls(name=data.get("name"))

        aperture_data = data.get("aperture")
        if aperture_data:
            optic.aperture = Aperture.from_dict(aperture_data)

        optic.surface_group = SurfaceGroup.from_dict(data["surface_group"])
        optic.wavelengths = WavelengthGroup.from_dict(data["wavelengths"])

        fields_data = data.get("fields")
        if fields_data:
            field_type_str = fields_data.get("field_type")
            if not field_type_str:
                raise ValueError("Field data is missing 'field_type' identifier.")

            optic.fields = make_field_group(field_type_str)
            optic.fields = optic.fields.from_dict(
                fields_data, optic_for_add_field=optic
            )
            optic.obj_space_telecentric = optic.fields.telecentric
        else:
            optic.fields = None
            # Ensure obj_space_telecentric is consistent if fields are None
            optic.obj_space_telecentric = data.get("object_space_telecentric", False)

        apodization_data = data.get("apodization")
        if apodization_data:
            # Assuming BaseApodization has a from_dict method
            optic.apodization = BaseApodization.from_dict(apodization_data)

        optic.pickups = PickupManager.from_dict(optic, data["pickups"])
        optic.solves = SolveManager.from_dict(optic, data["solves"])

        if "polarization" in data:
            optic.polarization = data["polarization"]
        elif "wavelengths" in data and "polarization" in data["wavelengths"]:
            # Legacy fallback
            optic.polarization = data["wavelengths"]["polarization"]
        else:
            optic.polarization = "ignore"

        # Final consistency check for obj_space_telecentric
        if optic.fields and optic.obj_space_telecentric != optic.fields.telecentric:
            # This indicates a potential mismatch if loaded from an older version
            # or if data is inconsistent. Prioritize field group's value.
            optic.obj_space_telecentric = optic.fields.telecentric
        elif "object_space_telecentric" in data and not optic.fields:
            optic.obj_space_telecentric = data["object_space_telecentric"]
        # If fields is None, obj_space_telecentric was set from data or
        # defaults to False

        optic.paraxial = Paraxial(optic)
        optic.aberrations = Aberrations(optic)
        optic.ray_tracer = RealRayTracer(optic)

        return optic
