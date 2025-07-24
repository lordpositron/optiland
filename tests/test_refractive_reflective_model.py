import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.materials import IdealMaterial
from optiland.rays import ParaxialRays, RealRays


class TestRefractiveReflectiveModel:
    def create_model(self, is_reflective=False):
        cs = CoordinateSystem()
        geometry = Plane(cs)
        material_pre = IdealMaterial(1.0)
        material_post = IdealMaterial(1.5)
        coating = SimpleCoating(0.5, 0.5)
        bsdf = None
        return RefractiveReflectiveModel(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            is_reflective=is_reflective,
            coating=coating,
            bsdf=bsdf,
        )

    def test_interact_real_rays_refraction(self, set_test_backend):
        model = self.create_model()
        rays = RealRays(be.zeros(1), be.zeros(1), be.zeros(1), be.zeros(1), be.zeros(1), be.ones(1), be.ones(1), be.ones(1))
        interacted_rays = model.interact_real_rays(rays)
        assert isinstance(interacted_rays, RealRays)
        # TODO: Add more specific assertions

    def test_interact_real_rays_reflection(self, set_test_backend):
        model = self.create_model(is_reflective=True)
        rays = RealRays(be.zeros(1), be.zeros(1), be.zeros(1), be.zeros(1), be.zeros(1), be.ones(1), be.ones(1), be.ones(1))
        interacted_rays = model.interact_real_rays(rays)
        assert isinstance(interacted_rays, RealRays)
        # TODO: Add more specific assertions

    def test_interact_paraxial_rays_refraction(self, set_test_backend):
        model = self.create_model()
        rays = ParaxialRays(be.ones(1), be.zeros(1), be.zeros(1), be.ones(1))
        interacted_rays = model.interact_paraxial_rays(rays)
        assert isinstance(interacted_rays, ParaxialRays)
        # TODO: Add more specific assertions

    def test_interact_paraxial_rays_reflection(self, set_test_backend):
        model = self.create_model(is_reflective=True)
        rays = ParaxialRays(be.ones(1), be.zeros(1), be.zeros(1), be.ones(1))
        interacted_rays = model.interact_paraxial_rays(rays)
        assert isinstance(interacted_rays, ParaxialRays)
        # TODO: Add more specific assertions
