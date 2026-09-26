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
MolMiner installs a light cheminformatics core by default. Additional modules can be installed if needed. 
```bash
pip install MolMiner
```

### Additional modules

| Install                           | Adds                                                                         | Usage                                                                                                                        |
|-----------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `pip install MolMiner`            | rdkit, pandas, datamol, scikit-learn, splito, umap-learn, matplotlib/seaborn | fingerprints, BRICS/RECAP fragmentation, molecule drawing, chemical space (PCA/t-SNE/UMAP), clustering, plots, ROC/PR curves |
| `pip install "MolMiner[ml]"`      | torch, keras, xgboost, optuna                                                | `mminer.models`, `mminer.hyperparams`                                                                                        |
| `pip install "MolMiner[docking]"` | prolif, meeko                                                                | `mminer.docking` interaction analysis and ligand prep, requires Autodock Vina installation using Conda or Pixi               |
| `pip install "MolMiner[safe]"`    | safe-mol, transformers                                                       | `slicer='safe'`, `SAFEbuild`, `Draw.safe_highlight`, `Matching`                                                              |
| `pip install "MolMiner[all]"`     | all of the above                                                             | everything except AutoDock Vina (see below)                                                                                  |

Installing additional modules at once: `pip install "MolMiner[ml,safe]"`.

Using a feature whose extra is missing reports what to install rather than failing with a bare
`ModuleNotFoundError`:

### Molecular Docking
Vina is **not** included in `[all]`, because it cannot be installed by pip on every platform. Installation of Vina 
should be done directly using Conda or Pixi before the associated MolMiner modules can be used.

## Quickstart
Tutorials will be forthcoming.