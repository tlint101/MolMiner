# MolMiner

[![MolMiner](https://img.shields.io/pypi/v/MolMiner.svg?label=MolMiner&style=flat)](https://pypi.org/project/MolMiner)
[![python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![scikit-learn](https://img.shields.io/badge/scikit-1.5.1-%23F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org/stable/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.1.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Keras](https://img.shields.io/badge/Keras-3.0.0+-%23D00000.svg?style=flat&logo=keras&logoColor=D00000)](https://keras.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MolMiner consolidates tools for small-molecule virtual screening. It includes custom scripts for data preprocessing,
molecular docking (Autodock Vina), and ML/DL model development (Scikit-Learn and Keras). Using MolMiner is optional and
was created to streamline environment setup without manually sourcing relevant Python project packages.

## Installation
MolMiner installs a light cheminformatics core by default. The heavier stacks are opt-in extras, so
you only pay for what you use.

```bash
pip install MolMiner
```

### Extras

| Install | Adds | Unlocks |
|---|---|---|
| `pip install MolMiner` | rdkit, pandas, datamol, scikit-learn, splito, umap-learn, matplotlib/seaborn | fingerprints, BRICS/RECAP fragmentation, molecule drawing, chemical space (PCA/t-SNE/UMAP), clustering, plots, ROC/PR curves |
| `pip install "MolMiner[ml]"` | torch, keras, xgboost, optuna | `mminer.models`, `mminer.hyperparams` |
| `pip install "MolMiner[docking]"` | prolif, meeko | `mminer.docking` interaction analysis and ligand prep |
| `pip install "MolMiner[safe]"` | safe-mol, transformers | `slicer='safe'`, `SAFEbuild`, `Draw.safe_highlight`, `Matching` |
| `pip install "MolMiner[all]"` | all of the above | everything except AutoDock Vina (see below) |

Extras combine: `pip install "MolMiner[ml,safe]"`.

Using a feature whose extra is missing reports what to install rather than failing with a bare
`ModuleNotFoundError`:

```
MissingExtraError: Keras requires the 'ml' extra, but 'keras' is not installed.
Install it with:
    pip install "MolMiner[ml]"
```

### AutoDock Vina
Vina is **not** included in `[all]`, because it cannot be installed by pip on every platform.
Published builds of `vina` 1.2.7:

| Channel | Platforms | Python |
|---|---|---|
| PyPI | linux x86_64 only | 3.8 - 3.12 |
| conda-forge | linux-64, linux-aarch64, linux-ppc64le, osx-64, osx-arm64 | 3.10 - 3.14 |

Neither channel ships Windows builds. Off these combinations pip falls back to compiling from
source against Boost and SWIG, which is the failure described in
[this issue](https://github.com/ccsb-scripps/AutoDock-Vina/issues/260).

**Plain venv** -- linux x86_64 with Python 3.11 or 3.12 only:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install "MolMiner[docking,vina]"
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

**Conda / mamba / pixi** -- required on macOS, Windows, or Python >= 3.13:
```bash
conda create -n molminer python=3.12
conda activate molminer
conda install -c conda-forge vina
pip install "MolMiner[docking]"
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

`AutoDockTools_py3` supplies `prepare_receptor4.py`, which `Vina.prep_receptor()` calls. It is pure
Python and installs the same way in any environment, but it can only be installed from git -- PyPI
does not accept direct-URL dependencies, so it can never be bundled into an extra.

## Quickstart
Tutorials will be forthcoming.