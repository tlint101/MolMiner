"""
Import modules for ML/AI modeling.
"""
from mminer._optional import lazy_exports

# Public name -> module that defines it. Modules load on first access so that a
# base install does not pay for extras it never uses.
_exports = {
    "Keras": "keras",
    "ml_results": "utils",
    "Models": "skmodels",
    "Lazy": "skmodels",
    "Splitter": "split",
    "CrossVal": "split",
}

__all__ = [
    "Keras",
    "ml_results",
    "Models",
    "Lazy",
    "Splitter",
    "CrossVal",
]

__getattr__, __dir__ = lazy_exports(__name__, _exports)
