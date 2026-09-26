"""
Initialize classes for features
"""
from mminer._optional import lazy_exports

# load modules
_exports = {
    "SAFEbuild": "build",
    "BRICSBuild": "build",
    "Score": "scoring",
    "Frag_Matching": "descriptions",
    "Fingerprint": "fingerprint",
    "Fragmentation": "fragmentation",
    "Matching": "matching",
    "sanitizer": "mol_prep",
}

__all__ = [
    "SAFEbuild",
    "BRICSBuild",
    "Score",
    "Frag_Matching",
    "Fingerprint",
    "Fragmentation",
    "Matching",
    "sanitizer",
]

__getattr__, __dir__ = lazy_exports(__name__, _exports)
