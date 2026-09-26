"""
Initialize classes for visualizer
"""
from mminer._optional import lazy_exports

# Public name -> module that defines it. Modules load on first access so that a
# base install does not pay for extras it never uses.
_exports = {
    "Cluster": "clustering",
    "MolecularFragments": "chemical_space",
    "Similarity": "chemical_space",
    "reduce_dimension": "chemical_space",
    "LibrarySpace": "chemical_space",
    "FragmentSpace": "chemical_space",
    "Analogs": "draw",
    "RDKitHighlight": "draw",
    "Draw": "draw",
    "ModelMetrics": "model_metrics",
    "ModelStats": "model_metrics",
    "ConfusionMatrix": "model_metrics",
    "ConfusionMatrixDisplay": "model_metrics",
    "ClassificationReport": "model_metrics",
    "Plots": "plots",
    "palette_hex": "utils",
}

__all__ = [
    "Cluster",
    "MolecularFragments",
    "Similarity",
    "reduce_dimension",
    "LibrarySpace",
    "FragmentSpace",
    "Analogs",
    "RDKitHighlight",
    "Draw",
    "ModelMetrics",
    "ModelStats",
    "ConfusionMatrix",
    "ConfusionMatrixDisplay",
    "ClassificationReport",
    "Plots",
    "palette_hex",
]

__getattr__, __dir__ = lazy_exports(__name__, _exports)
