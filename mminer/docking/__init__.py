"""
Initialize classes for docking.
"""
from mminer._optional import lazy_exports

# Public name -> module that defines it. Modules load on first access so that a
# base install does not pay for extras it never uses.
_exports = {
    "Vina": "autodock",
    "calculate_rmsd": "utils",
    "plot_3d": "utils",
}

__all__ = [
    "Vina",
    "calculate_rmsd",
    "plot_3d",
]

__getattr__, __dir__ = lazy_exports(__name__, _exports)
