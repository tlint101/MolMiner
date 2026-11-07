"""
Initialize classes for hyperparams
"""
from mminer._optional import lazy_exports

# Public name -> module that defines it. Modules load on first access so that a
# base install does not pay for extras it never uses.
_exports = {
    "Optimization": "optimization",
    "EarlyStopping": "optimization",
    "Sklearn_opt": "sklearn_opt",
    "Keras_opt": "keras_opt",
}

__all__ = [
    "Optimization",
    "EarlyStopping",
    "Sklearn_opt",
    "Keras_opt",
]

__getattr__, __dir__ = lazy_exports(__name__, _exports)

if __name__ == "__main__":
    import doctest

    doctest.testmod()

