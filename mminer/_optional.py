"""
Helpers for importing optional dependencies that live behind a pip extra.

MolMiner defaults to a light base install. Heavier dependencies are opt-in extras. Modules will be imported lazily
through :func:`require`, so a base install can still use everything that does not need them.
"""
import sys
import importlib
from io import StringIO
from typing import Optional

__all__ = ["require", "lazy_exports", "MissingExtraError"]

# Which extra provides which distribution, for the error message.
_EXTRA_FOR_MODULE = {
    "safe": "safe",
    "transformers": "safe",
    "torch": "ml",
    "keras": "ml",
    "xgboost": "ml",
    "optuna": "ml",
    "meeko": "docking",
}

# Packages that must be installed from conda-forge (via Pixi or Conda) rather than PyPI.
_CONDA_ONLY_INSTALL = {
    "vina": (
        "'vina' lacks PyPI wheels on macOS, Windows, and Python >= 3.13.\n"
        "Install it from conda-forge using Pixi or Conda:\n"
        "    pixi add vina\n"
        "    # or\n"
        "    conda install -c conda-forge vina"
    ),
}


class MissingExtraError(ImportError):
    """Raised when an optional dependency is used without its extra installed."""


def _missing_extra_error(module: str, extra: Optional[str] = None,
                         feature: Optional[str] = None) -> "MissingExtraError":
    """Build the install-hint error for a missing optional dependency."""
    feature = feature or module
    if module in _CONDA_ONLY_INSTALL:
        return MissingExtraError(
            f"{feature} requires '{module}', which is not installed.\n"
            f"{_CONDA_ONLY_INSTALL[module]}"
        )
    extra = extra or _EXTRA_FOR_MODULE.get(module, module)
    return MissingExtraError(
        f"{feature} requires the '{extra}' extra, but '{module}' is not installed.\n"
        f"Install it with:\n"
        f"    pip install \"MolMiner[{extra}]\""
    )


def require(module: str, extra: Optional[str] = None, feature: Optional[str] = None):
    """
    Import an optional dependency, muting its import-time noise.

    Some dependencies (notably safe-mol and umap-learn) print to stdout/stderr
    while importing. Those streams are swapped for throwaway buffers during the
    import and restored in a ``finally`` block, so a failed import can never
    leave the caller's terminal muted.

    :param module: Module name to import, e.g. ``"safe"``.
    :param extra: Pip extra that provides it. Defaults to a built-in lookup.
    :param feature: Human-readable name of the calling feature, used in the
        error message. Defaults to the module name.
    :return: The imported module.
    :raises MissingExtraError: If the module is not installed.

    :Example:
        >>> sf = require("safe", "safe", "SAFE fragmentation")  # doctest: +SKIP
    """
    extra = extra or _EXTRA_FOR_MODULE.get(module, module)
    feature = feature or module

    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = StringIO(), StringIO()
    try:
        return importlib.import_module(module)
    except ImportError as error:
        # A genuine error inside an installed package should surface as-is.
        if getattr(error, "name", None) not in (module, module.split(".")[0]):
            raise
        raise _missing_extra_error(module, extra, feature) from error
    finally:
        sys.stdout, sys.stderr = _stdout, _stderr


def lazy_exports(package: str, attr_map: dict):
    """
    Build ``__getattr__``/``__dir__`` for a subpackage that loads modules on demand.

    Importing a subpackage should not drag in every optional dependency its
    modules use, so the public names resolve lazily. A module that fails because
    its extra is not installed reports the install hint rather than a bare
    ``ModuleNotFoundError``.

    :param package: Importing package name, i.e. ``__name__``.
    :param attr_map: Mapping of public name -> submodule that defines it.
    :return: ``(__getattr__, __dir__)`` to assign in the package's ``__init__``.
    """

    def __getattr__(name: str):
        module_name = attr_map.get(name)
        if module_name is None:
            raise AttributeError(f"Module '{package}' has no attribute '{name}'")
        try:
            module = importlib.import_module(f"{package}.{module_name}")
        except ImportError as error:
            missing = (getattr(error, "name", None) or "").split(".")[0]
            if missing in _EXTRA_FOR_MODULE or missing in _CONDA_ONLY_INSTALL:
                raise _missing_extra_error(missing, feature=name) from error
            raise
        value = getattr(module, name)
        # Cache on the package so repeat lookups skip __getattr__ entirely.
        setattr(sys.modules[package], name, value)
        return value

    def __dir__():
        return sorted(attr_map)

    return __getattr__, __dir__


if __name__ == "__main__":
    import doctest

    doctest.testmod()
