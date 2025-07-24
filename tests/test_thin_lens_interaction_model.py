import pytest
import optiland.backend as be
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.rays import RealRays, ParaxialRays
from optiland.coordinate_system import CoordinateSystem


@pytest.fixture
def thin_lens_interaction_model():
    """Fixture for a ThinLensInteractionModel."""
    return ThinLensInteractionModel(
        focal_length=10.0,
        geometry=Plane(CoordinateSystem()),
        material_pre=IdealMaterial(n=1.0),
        material_post=IdealMaterial(n=1.5),
        is_reflective=False
    )


def test_interact_real_rays(thin_lens_interaction_model):
    """Test interact_real_rays method of ThinLensInteractionModel."""
    rays = RealRays(
        x=be.array([1.0]),
        y=be.array([1.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        w=be.array([0.55]),
        i=be.array([1.0])
    )
    interacted_rays = thin_lens_interaction_model.interact_real_rays(rays)
    assert interacted_rays is not None
    assert isinstance(interacted_rays, RealRays)


def test_interact_paraxial_rays(thin_lens_interaction_model):
    """Test interact_paraxial_rays method of ThinLensInteractionModel."""
    rays = ParaxialRays(
        y=be.array([1.0]),
        u=be.array([0.1]),
        w=be.array([0.55]),
        z=be.array([0.0])
    )
    interacted_rays = thin_lens_interaction_model.interact_paraxial_rays(rays)
    assert interacted_rays is not None
    assert isinstance(interacted_rays, ParaxialRays)
