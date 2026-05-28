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
### PyPI
Installation of MolMiner is done as follows:
```bash
pip install MolMiner
```
During testing using pip, Autodock Vina could not be installed. This is a pip install problem ([issue here](https://github.com/ccsb-scripps/AutoDock-Vina/issues/260)). 
As a result, **the Autodock Vina package is not a part of the MolMiner when installed using pip**. Users who want docking
should install it using Conda.

### Conda
```bash
conda create -n MolMiner
conda activate molminer
pip install MolMiner
```
Then install Autodock Vina and AutoDockTools_py3 scripts:
```bash
conda install -c conda-forge vina
pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

## Quickstart
Tutorials will be forthcoming.