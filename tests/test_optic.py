import pytest

import optiland.backend as be
from optiland.aperture import Aperture
from optiland.apodization import GaussianApodization
from optiland.fields.field_group import FieldGroup
from optiland.optic import Optic
from optiland.rays import create_polarization
from optiland.samples.objectives import HeliarLens
from optiland.surfaces import SurfaceGroup
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.wavelength import WavelengthGroup
from tests.utils import assert_allclose


def singlet_infinite_object():
    lens = Optic()
    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.add_surface(index=2, radius=-46.2795, thickness=50)
    lens.add_surface(index=3)

    lens.set_aperture(aperture_type="EPD", value=25)

    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)

    lens.add_wavelength(value=0.5, is_primary=True)

    return lens


def singlet_finite_object():
    lens = Optic()
    lens.add_surface(index=0, radius=be.inf, thickness=50)
    lens.add_surface(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.add_surface(index=2, radius=-46.2795, thickness=50)
    lens.add_surface(index=3)

    lens.set_aperture(aperture_type="EPD", value=25)

    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)

    lens.add_wavelength(value=0.5, is_primary=True)

    return lens


class TestOptic:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Optic()

    def test_initialization(self, set_test_backend):
        assert self.optic.aperture is None
        assert self.optic.field_type is None
        assert isinstance(self.optic.surface_group, SurfaceGroup)
        assert isinstance(self.optic.fields, FieldGroup)
        assert isinstance(self.optic.wavelengths, WavelengthGroup)
        assert self.optic.polarization == "ignore"
        assert not self.optic.obj_space_telecentric

    def test_add_surface(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        assert len(self.optic.surface_group.surfaces) == 1

    def test_add_field(self, set_test_backend):
        # Minimal setup for set_field_type to pass validation.
        # Ensure object surface is finite for "object_height", or use "angle".
        # Finite object by default due to thickness=10.
        self.optic.add_surface(index=0, radius=be.inf, thickness=10)
        self.optic.add_surface(index=1, radius=be.inf, thickness=0)  # Image

        self.optic.set_field_type("object_height")
        self.optic.add_field(10.0, 5.0)
        assert len(self.optic.fields.fields) == 1
        current_field = self.optic.fields.fields[0]
        assert current_field.y == 10.0
        assert current_field.x == 5.0
        assert current_field.field_type == "object_height"

        # Test with angle type
        self.optic.set_field_type("angle")
        self.optic.add_field(2.0, 1.0)
        assert len(self.optic.fields.fields) == 2 # Now two fields
        current_field_angle = self.optic.fields.fields[1]
        assert current_field_angle.y == 2.0
        assert current_field_angle.x == 1.0
        assert current_field_angle.field_type == "angle"


    def test_add_wavelength(self, set_test_backend):
        self.optic.add_wavelength(0.55, is_primary=True)
        assert len(self.optic.wavelengths.wavelengths) == 1
        assert self.optic.wavelengths.wavelengths[0].value == 0.55
        assert self.optic.wavelengths.wavelengths[0].is_primary

    def test_set_aperture(self, set_test_backend):
        self.optic.set_aperture("EPD", 5.0)
        assert isinstance(self.optic.aperture, Aperture)
        assert self.optic.aperture.ap_type == "EPD"
        assert self.optic.aperture.value == 5.0

    def test_set_field_type(self, set_test_backend):
        from optiland.fields.strategies import (
            AngleField,
            ObjectHeightField,
        )

        # Minimal optic setup for validation
        self.optic.add_surface(index=0, radius=be.inf, thickness=10) # Finite obj
        self.optic.add_surface(index=1, radius=be.inf, thickness=0)  # Image

        self.optic.set_field_type("angle")
        assert isinstance(self.optic.field_type, AngleField)

        # Ensure optic state is valid for "object_height" (e.g., finite object).
        # The current setup (thickness=10 for surface 0) makes it finite.
        if self.optic.object_surface:
             self.optic.object_surface.is_infinite = False # Explicitly finite

        self.optic.set_field_type("object_height")
        assert isinstance(self.optic.field_type, ObjectHeightField)


    def test_set_comment(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
            comment="Object surface",
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="SF11",
            thickness=5,
            comment="First surface",
        )

        assert self.optic.surface_group.surfaces[0].comment == "Object surface"
        assert self.optic.surface_group.surfaces[1].comment == "First surface"

    def test_set_radius(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.set_radius(10.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 10.0

    def test_set_conic(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.set_conic(-1.0, 0)
        assert self.optic.surface_group.surfaces[0].geometry.k == -1.0

    def test_set_thickness(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.add_surface(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        self.optic.set_thickness(10.0, 1)
        assert self.optic.surface_group.get_thickness(1) == 10.0

    def test_set_index(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        self.optic.set_index(1.5, 1)
        assert self.optic.surface_group.surfaces[1].material_post.n(1) == 1.5

    def test_set_material(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.add_surface(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        surface_number = 1
        material_post = MaterialFactory._configure_post_material('N-BK7')
        self.optic.set_material(material_post, surface_number)
        surface = self.optic.surface_group.surfaces[surface_number]
        assert surface.material_post == material_post

    def test_set_asphere_coeff(self, set_test_backend):
        self.optic.add_surface(
            index=0,
            surface_type="even_asphere",
            material="air",
            thickness=5,
            coefficients=[0.0, 0.0, 0.0],
        )
        self.optic.set_asphere_coeff(0.1, 0, 2)
        assert self.optic.surface_group.surfaces[0].geometry.c[2] == 0.1

    def test_set_polarization(self, set_test_backend):
        self.optic.set_polarization("ignore")
        assert self.optic.polarization == "ignore"

    def test_optic_default_apodization(self, set_test_backend):
        assert self.optic.apodization is None

    def test_optic_set_apodization(self, set_test_backend):
        gaussian_apod = GaussianApodization(sigma=0.5)
        self.optic.set_apodization(gaussian_apod)
        assert self.optic.apodization == gaussian_apod

        with pytest.raises(TypeError):
            self.optic.set_apodization("not_an_apodization_object")

    def test_set_invalid_polarization(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.set_polarization("invalid")

    def test_set_pickup(self, set_test_backend):
        self.optic.add_surface(index=0, surface_type="standard", thickness=5)
        self.optic.add_surface(index=1, surface_type="standard", thickness=5)
        self.optic.pickups.add(0, "radius", 1, scale=2, offset=1)
        assert len(self.optic.pickups) == 1

    def test_clear_pickups(self, set_test_backend):
        self.optic.add_surface(index=0, surface_type="standard", thickness=5)
        self.optic.add_surface(index=1, surface_type="standard", thickness=5)
        self.optic.pickups.add(0, "radius", 1, scale=2, offset=1)
        self.optic.pickups.clear()
        assert len(self.optic.pickups) == 0

    def test_set_solve(self, set_test_backend):
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        assert len(optic.solves) == 1

    def test_clear_solves(self, set_test_backend):
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        optic.solves.clear()
        assert len(optic.solves) == 0

    def test_scale_system(self, set_test_backend):
        self.optic.add_surface(index=0, radius=10, thickness=5)
        self.optic.set_aperture("EPD", 5.0)
        self.optic.scale_system(2)
        assert self.optic.surface_group.surfaces[0].geometry.radius == 2 * 10.0
        assert self.optic.aperture.value == 2 * 5.0

    def test_reset(self, set_test_backend):
        self.optic.add_surface(index=0, radius=10, thickness=5)
        self.optic.reset()
        assert self.optic.aperture is None
        assert self.optic.field_type is None
        assert len(self.optic.surface_group.surfaces) == 0
        assert len(self.optic.fields.fields) == 0
        assert len(self.optic.wavelengths.wavelengths) == 0
        assert len(self.optic.pickups) == 0
        assert len(self.optic.solves) == 0

    def test_n(self, set_test_backend):
        self.optic.add_surface(index=0, radius=10, thickness=5)
        self.optic.add_wavelength(0.55, is_primary=True)
        n_values = self.optic.n()
        assert len(n_values) == 1

    def test_update_paraxial(self, set_test_backend):
        lens = HeliarLens()
        lens.update_paraxial() # Ensures method runs without error

    def test_update(self, set_test_backend):
        lens = HeliarLens()
        lens.update() # Ensures method runs without error

    def test_image_solve(self, set_test_backend):
        lens = HeliarLens()
        lens.image_solve() # Ensures method runs without error

    def test_trace(self, set_test_backend):
        lens = HeliarLens()
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_generic(self, set_test_backend):
        lens = HeliarLens()
        rays = lens.trace_generic(0.0, 0.0, 0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_invalid_field(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError): # Field out of bounds
            lens.trace(0.0, 2.0, 0.55)

    def test_trace_generic_invalid_field(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError): # Field out of bounds
            lens.trace_generic(0.0, 0.0, 0.0, 2.0, 0.55)

    def test_trace_generic_invalid_pupil(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError): # Pupil out of bounds
            lens.trace_generic(0.0, 5.0, 0.0, 0.0, 0.55)

    def test_trace_polarized(self, set_test_backend):
        lens = HeliarLens()
        state = create_polarization("unpolarized")
        lens.set_polarization(state)
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_object_property(self, set_test_backend):
        assert self.optic.object_surface is None

    def test_image_surface_property(self, set_test_backend):
        self.optic.add_surface(index=0, radius=10, thickness=5)
        assert self.optic.image_surface is self.optic.surface_group.surfaces[0]

    def test_total_track_property(self, set_test_backend):
        lens = HeliarLens()
        assert lens.total_track == 12.1357

    def test_total_track_error(self, set_test_backend):
        lens = HeliarLens()
        lens.surface_group.surfaces = [lens.surface_group.surfaces[0]] # Invalid
        with pytest.raises(ValueError):
            _ = lens.total_track

    def test_polarization_state_property(self, set_test_backend):
        lens = HeliarLens()
        assert lens.polarization_state is None

        state = create_polarization("unpolarized")
        lens.set_polarization(state)
        assert lens.polarization_state == state

    def test_polarization_state_error(self, set_test_backend):
        lens = HeliarLens()
        lens.polarization = "invalid_state_string" # Invalid direct assignment
        with pytest.raises(ValueError):
            _ = lens.polarization_state

    def test_to_dict(self, set_test_backend):
        lens = HeliarLens() # Default field_type is "angle"
        lens.set_apodization(GaussianApodization(sigma=0.5))
        lens_dict = lens.to_dict()
        assert isinstance(lens_dict, dict)
        assert "apodization" in lens_dict
        assert lens_dict["apodization"]["type"] == "GaussianApodization"
        assert lens_dict["apodization"]["sigma"] == 0.5
        assert "fields" in lens_dict
        assert lens_dict["fields"]["field_type"] == "angle"

    def test_from_dict(self, set_test_backend):
        from optiland.fields.strategies import AngleField, ObjectHeightField

        lens = HeliarLens() # field_type "angle"
        lens.set_apodization(GaussianApodization(sigma=0.5))
        basic_dict = lens.to_dict()

        new_optic = Optic.from_dict(basic_dict)
        assert isinstance(new_optic, Optic)
        assert isinstance(new_optic.apodization, GaussianApodization)
        assert isinstance(new_optic.field_type, AngleField)

        basic_dict_no_apod = lens.to_dict()
        basic_dict_no_apod.pop("apodization", None)
        new_optic_no_apod = Optic.from_dict(basic_dict_no_apod)
        assert new_optic_no_apod.apodization is None
        assert isinstance(new_optic_no_apod.field_type, AngleField)

        finite_lens = singlet_finite_object() # Default field_type "angle"
        # Change to object_height for testing this case
        finite_lens.set_field_type("object_height")
        finite_dict_obj_height = finite_lens.to_dict()
        assert finite_dict_obj_height["fields"]["field_type"] == "object_height"

        new_optic_obj_height = Optic.from_dict(finite_dict_obj_height)
        assert isinstance(new_optic_obj_height.field_type, ObjectHeightField)

        dict_missing_ft = lens.to_dict()
        dict_missing_ft["fields"].pop("field_type", None)
        # Optic.from_dict should raise ValueError if fields are present
        # but field_type string is missing.
        if new_optic.fields.fields: # Heliar lens has fields
             with pytest.raises(ValueError) as excinfo:
                 Optic.from_dict(dict_missing_ft)
             assert "missing in dictionary but fields are present" in str(excinfo.value)

        dict_empty_ft = lens.to_dict()
        dict_empty_ft["fields"]["field_type"] = ""
        if new_optic.fields.fields:
            with pytest.raises(ValueError) as excinfo:
                Optic.from_dict(dict_empty_ft)
            assert "missing in dictionary but fields are present" in str(excinfo.value)

        empty_optic = Optic()
        empty_dict = empty_optic.to_dict() # field_type will be ""
        empty_dict["fields"]["field_type"] = ""
        deserialized_empty_optic = Optic.from_dict(empty_dict)
        assert deserialized_empty_optic.field_type is None
        assert not deserialized_empty_optic.fields.fields


    def test_set_invalid_field_type_string(self, set_test_backend):
        with pytest.raises(ValueError) as excinfo:
            self.optic.set_field_type("invalid_field_type_name")
        assert "Invalid field_type string" in str(excinfo.value)

    def test_validate_optic_state_object_height_infinite_obj(self, set_test_backend):
        self.optic.add_surface(index=0, thickness=be.inf) # Infinite object
        self.optic.add_surface(index=1, thickness=0)   # Image surface
        with pytest.raises(ValueError) as excinfo:
            self.optic.set_field_type("object_height")
        err_msg = 'Field type "object_height" is invalid for an object at infinity.'
        assert err_msg in str(excinfo.value)

    def test_validate_optic_state_angle_telecentric(self, set_test_backend):
        self.optic.add_surface(index=0, thickness=10) # Finite object
        self.optic.add_surface(index=1, thickness=0)  # Image surface
        self.optic.obj_space_telecentric = True
        with pytest.raises(ValueError) as excinfo:
            self.optic.set_field_type("angle")
        err_msg = 'Field type cannot be "angle" for telecentric object space.'
        assert err_msg in str(excinfo.value)
        self.optic.obj_space_telecentric = False

    def test_validate_optic_state_object_height_telecentric_epd(self, set_test_backend):
        self.optic.add_surface(index=0, thickness=10) # Finite object
        self.optic.add_surface(index=1, thickness=0)  # Image surface
        self.optic.obj_space_telecentric = True
        self.optic.set_aperture("EPD", 10)
        with pytest.raises(ValueError) as excinfo:
            self.optic.set_field_type("object_height")
        err_msg = 'Aperture type cannot be "EPD" for telecentric object space'
        assert err_msg in str(excinfo.value)
        self.optic.obj_space_telecentric = False

    def test_validate_optic_state_object_height_telecentric_fno(self, set_test_backend):
        self.optic.add_surface(index=0, thickness=10) # Finite object
        self.optic.add_surface(index=1, thickness=0)  # Image surface
        self.optic.obj_space_telecentric = True
        self.optic.set_aperture("imageFNO", 5)
        with pytest.raises(ValueError) as excinfo:
            self.optic.set_field_type("object_height")
        err_msg = 'Aperture type cannot be "imageFNO" for telecentric object space'
        assert err_msg in str(excinfo.value)
        self.optic.obj_space_telecentric = False

    # Test successful setting of field types is covered by
    # the modified test_set_field_type.

    def test_no_stop(self, set_test_backend):
        # Ensure optic has surfaces but none are stop
        self.optic.add_surface(index=0, thickness=10)
        self.optic.add_surface(index=1, thickness=10)
        for surface in self.optic.surface_group.surfaces:
            surface.is_stop = False # Explicitly set all to not be stop
        with pytest.raises(ValueError):
            _ = self.optic.surface_group.stop_index

    def test_add_infinite_object(self):
        lens1 = singlet_infinite_object()
        lens2 = singlet_infinite_object()
        lens_combined = lens1 + lens2
        assert lens_combined.surface_group.num_surfaces == 6

        rays = lens_combined.trace(
            Hx=0, Hy=0, distribution="random", num_rays=42, wavelength=0.5
        )
        assert rays is not None

    def test_add_finite_object(self):
        lens1 = singlet_finite_object()
        lens2 = singlet_finite_object()
        lens_combined = lens1 + lens2
        assert lens_combined.surface_group.num_surfaces == 6

        rays = lens_combined.trace(
            Hx=0, Hy=0, distribution="random", num_rays=42, wavelength=0.5
        )
        assert rays is not None

    def test_invalid_coordinate_system(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.add_surface(index=0, radius=be.inf, z=-100)
            # Cannot use dx or dy with absolute z positioning
            self.optic.add_surface(index=1, radius=be.inf, z=0, dx=15)

    def test_flip_optic(self, set_test_backend):
        lens = HeliarLens()
        lens.surface_group.set_fresnel_coatings()
        radii_orig = be.copy(lens.surface_group.radii)
        radii_orig = radii_orig[~be.isinf(radii_orig)]
        n_orig = be.copy(lens.n(0.55))
        lens.flip()
        radii_flipped = be.copy(lens.surface_group.radii)
        radii_flipped = radii_flipped[~be.isinf(radii_flipped)]
        n_flipped = be.copy(lens.n(0.55))
        assert_allclose(radii_orig, -be.flip(radii_flipped))
        assert_allclose(n_orig[:-1], be.flip(n_flipped[:-1]))

    def test_invalid_flip(self, set_test_backend):
        lens = Optic() # Optic with no surfaces or only 1 surface
        with pytest.raises(ValueError):
            lens.flip()

    def test_flip_solves_pickups(self, set_test_backend):
        lens = Optic()

        lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        lens.add_surface(index=1, radius=100, thickness=4,
                         material="SK16", is_stop=True)
        lens.add_surface(index=2, radius=-1000, thickness=20)
        lens.add_surface(index=3) # Image surface
        lens.set_aperture(aperture_type="EPD", value=10.0)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        lens.add_wavelength(value=0.5876, is_primary=True)

        lens.solves.add("quick_focus")
        lens.pickups.add(
            source_surface_idx=1,
            attr_type="radius",
            target_surface_idx=2,
            scale=-1,
            offset=0,
        )
        lens.update()
        lens.flip()
        assert lens.pickups.pickups[0].source_surface_idx == 2
        assert lens.pickups.pickups[0].target_surface_idx == 1
