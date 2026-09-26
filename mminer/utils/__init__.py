"""
Initialize classes. Script added to simplify import.
"""
from mminer._optional import lazy_exports

# Public name -> module that defines it. Modules load on first access so that a
# base install does not pay for extras it never uses.
_exports = {
    "ButinaClusters": "cluster_utils",
    "UmapClusters": "cluster_utils",
    "smiles_to_fp": "fingerprint_misc",
}

__all__ = [
    "ButinaClusters",
    "UmapClusters",
    "smiles_to_fp",
]

__getattr__, __dir__ = lazy_exports(__name__, _exports)
