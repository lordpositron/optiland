from abc import ABC, abstractmethod
from typing import Optional

from optiland.rays import ParaxialRays, RealRays
from optiland.geometries import BaseGeometry
from optiland.materials import BaseMaterial
from optiland.coatings import BaseCoating
from optiland.scatter import BaseBSDF


class BaseInteractionModel(ABC):
    """Abstract base class for ray-surface interaction models."""

    def __init__(
        self,
        geometry: BaseGeometry,
        material_pre: BaseMaterial,
        material_post: BaseMaterial,
        is_reflective: bool,
        coating: Optional[BaseCoating] = None,
        bsdf: Optional[BaseBSDF] = None,
    ):
        self.geometry = geometry
        self.material_pre = material_pre
        self.material_post = material_post
        self.is_reflective = is_reflective
        self.coating = coating
        self.bsdf = bsdf

    @abstractmethod
    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interact with real rays."""
        pass

    @abstractmethod
    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays."""
        pass

    def _apply_coating_and_bsdf(self, rays: RealRays, nx: float, ny: float, nz: float) -> RealRays:
        """Apply coating and BSDF to the rays."""
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        if self.coating:
            rays = self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        else:
            rays.update()
        return rays
