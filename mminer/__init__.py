"""
Authors: Tony E. Lin

Toolsets for the MolMiner.
"""
import warnings
import sys
from io import StringIO
import importlib

# suppress warnings from safe and umap
warnings.filterwarnings("ignore", message=r".*inputs is part of.*")
warnings.filterwarnings("ignore", module="umap", category=ImportWarning)

# temporarily mute stdout/stderr
_original_stdout, _original_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = StringIO(), StringIO()

# 3. Trigger eager imports that generate the warnings
# 
import safe  # noqa: F401
import umap  # noqa: F401

# 4. Restore streams immediately; keep warning filters active for downstream imports
sys.stdout, sys.stderr = _original_stdout, _original_stderr

_submodules = ["classification", "docking", "features", "hyperparams", "models", "utils", "visualizer"]

def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f"mminer.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'mminer' has no attribute '{name}'")
