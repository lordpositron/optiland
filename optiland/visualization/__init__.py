from . import coloring
from . import lens
from . import mirror
from . import rays
from . import surface
from . import system
from . import utils
from .visualization import (
    OpticViewer,
    OpticViewer3D,
    LensInfoViewer,
)  # Import specific classes

__all__ = [
    "coloring",
    "lens",
    "mirror",
    "rays",
    "surface",
    "system",
    "utils",
    "visualization",  # keep the module itself available if needed
    "OpticViewer",  # export the class
    "OpticViewer3D",  # export the class
    "LensInfoViewer",  # export the class
]
