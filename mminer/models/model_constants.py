"""
Script to hold default model and their parameter values
"""
from collections import namedtuple
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import numpy as np
import random

# To reduce the max_iter warning with LogisticRegression()
import warnings
from sklearn.exceptions import ConvergenceWarning

# set seed
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module="sklearn.linear_model.LogisticRegression",
)

"""Default Classification param grids"""

lr_param_grid = {
    "max_iter": (800, 3000),
    "C": (0.01, 100.0, "log"),
    # each solver supports different l1_ratio range, so solver must be conditional
    # convert solve into dictionary
    "solver": {
        "saga": {"l1_ratio": (0.0, 1.0)},
        "liblinear": {"l1_ratio": (0.0, 1.0, 1.0), "intercept_scaling": (1, 10)},  # L2 or L1 only
        # L2 only, the default l1_ratio=0
        "lbfgs": {},
        "newton-cg": {},
        "newton-cholesky": {},
        "sag": {},
    },
    "tol": (0.000001, 0.0001),
    "class_weight": [None, "balanced"],
    "random_state": [random_seed],
}

rf_param_grid = {
    "n_estimators": (10, 500),
    "max_depth": (1, 30),
    "min_samples_split": (2, 15),
    "min_samples_leaf": (1, 10),
    "max_features": [None, "sqrt", "log2"],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    "random_state": [random_seed],
    "n_jobs": [-1]
}

svc_param_grid = {
    "max_iter": 10000,
    "C": (0.1, 10.0),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": (0.1, 20),
    "degree": (1, 5),  # param only used with poly
    "class_weight": [None, "balanced"],
    "tol": (0.00001, 0.01),
    "random_state": [random_seed],
}

knn_param_grid = {
    "n_neighbors": (1, 100),
    "weights": [None, "uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
    "n_jobs": [-1]
}

xgb_param_grid = {
    "max_depth": (2, 20),
    "n_estimators": (5, 200),
    "eta": (0.001, 0.8),
    "min_child_weight": (0, 20),
    "subsample": (0.1, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "objective": ["binary:logistic"],
    "max_delta_step": [0, 1],
    "scale_pos_weight": (1, 10),
    "random_state": [random_seed],
    "n_jobs": [-1]
}

"""Default Regression param grids"""
ridge_param_grid = {
    "alpha": (0.001, 100.0, "log"),
    "fit_intercept": [True, False],
    "random_state": [random_seed],
}

enet_param_grid = {
    "max_iter": (1000, 5000),
    "alpha": (0.0001, 10.0, "log"),
    "l1_ratio": (0.01, 1.0),  # l1_ratio=0 (pure L2) does not converge well, use ridge instead
    "selection": ["cyclic", "random"],
    "tol": (0.00001, 0.001),
    "random_state": [random_seed],
}

rf_reg_param_grid = {
    "n_estimators": (10, 500),
    "max_depth": (1, 30),
    "min_samples_split": (2, 15),
    "min_samples_leaf": (1, 10),
    "max_features": [None, "sqrt", "log2"],
    "bootstrap": [True, False],
    "criterion": ["squared_error", "absolute_error"],
    "random_state": [random_seed],
    "n_jobs": [-1]
}

svr_param_grid = {
    "max_iter": 10000,
    "C": (0.1, 10.0),
    "epsilon": (0.01, 1.0, "log"),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": (0.1, 20),
    "degree": (1, 5),  # param only used with poly
    "tol": (0.00001, 0.01),
}

xgb_reg_param_grid = {
    "max_depth": (2, 20),
    "n_estimators": (5, 200),
    "eta": (0.001, 0.8),
    "min_child_weight": (0, 20),
    "subsample": (0.1, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "objective": ["reg:squarederror"],
    "random_state": [random_seed],
    "n_jobs": [-1]
}

PARAM = namedtuple("PARAM", ["name", "param_grid", "estimator", "full_name"])
PARAM_LIST = [
    PARAM("rf", rf_param_grid, RandomForestClassifier, "Random Forest"),
    PARAM("svc", svc_param_grid, SVC, "C-Support Vector"),
    PARAM("lr", lr_param_grid, LogisticRegression, "Logistic Regression"),
    PARAM("knn", knn_param_grid, KNeighborsClassifier, "K-Nearest Neighbors"),
    PARAM("xgb", xgb_param_grid, XGBClassifier, "XGBoost"),
]

REG_PARAM_LIST = [
    PARAM("ridge", ridge_param_grid, Ridge, "Ridge Regression"),
    PARAM("enet", enet_param_grid, ElasticNet, "Elastic Net"),
    PARAM("rf", rf_reg_param_grid, RandomForestRegressor, "Random Forest"),
    PARAM("knn", knn_param_grid, KNeighborsRegressor, "K-Nearest Neighbors"),
    PARAM("svr", svr_param_grid, SVR, "Support Vector"),
    PARAM("xgb", xgb_reg_param_grid, XGBRegressor, "XGBoost"),
]

# task -> default param grids
PARAM_TASKS = {
    "classification": PARAM_LIST,
    "regression": REG_PARAM_LIST,
}
