"""
Authors: Tony E. Lin

Toolsets for the MolMiner.
"""
import warnings
import importlib

# suppress warnings from safe and umap
warnings.filterwarnings("ignore", message=r".*inputs is part of.*")
warnings.filterwarnings("ignore", module="umap", category=ImportWarning)

_submodules = ["classification", "docking", "features", "hyperparams", "logger", "models", "utils", "visualizer"]


def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f"mminer.{name}")
    raise AttributeError(f"Module 'mminer' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | set(_submodules))
