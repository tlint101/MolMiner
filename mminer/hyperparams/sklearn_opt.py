"""
Script to run sklearn and xgb hyperparameter optimizations.
"""
import pandas as pd
import numpy as np
import json
import optuna
from optuna.samplers import TPESampler
from typing import Optional, Union, Iterable, Callable
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, matthews_corrcoef
from xgboost import XGBClassifier
from mminer.models.skmodels import Models
from mminer.models.split import CrossVal
from mminer.models.model_constants import PARAM_LIST
from mminer.hyperparams.optimization import Optimization


# For XGBoost reference
# https://www.kaggle.com/code/mh0386/facial-emotions-detection#SVM


class Sklearn_opt(Optimization):
    def __init__(self,
                 x_train: Union[np.ndarray, pd.DataFrame] = None,
                 x_test: Union[np.ndarray, pd.DataFrame] = None,
                 y_train: Union[np.ndarray, pd.DataFrame] = None,
                 y_test: Union[np.ndarray, pd.DataFrame] = None,
                 features: Union[np.ndarray, pd.DataFrame] = None,
                 labels: Union[np.ndarray, pd.DataFrame] = None,
                 model: Union[str, BaseEstimator] = None,
                 cv: Optional[int] = None,
                 precision_weight: float = 0,
                 mcc_weight: float = 0
                 ):
        """
        Initialize the Sklearn_opt clas
        :param x_train: Union[np.ndarray, pd.DataFrame]
            Input the training features for the given problem.
        :param x_test: Union[np.ndarray, pd.DataFrame]
            Input the testing features for the given problem.
        :param y_train: Union[np.ndarray, pd.DataFrame]
            Input the training labels for the given problem.
        :param y_test: Union[np.ndarray, pd.DataFrame]
            Input the testing labels for the given problem.
        :param features: Union[np.ndarray, pd.DataFrame]
            Input the features for the entire dataset.
        :param labels: Union[np.ndarray, pd.DataFrame]
            Input the labels for the entire dataset.
        :param model: Union[str, BaseEstimator]
            Set model to optimize. If none given, models can be called using the class methods.
        :param cv: int
            Set the number of cv folds. found with the dataset. The features/labels/train/test sets should have the
            corresponding cv folds labeled.
        :param precision_weight: float
            Set the weight for precision score. The value should be between 0 and 1. The corresponding weight for
            accuracy will be calculated as 1 - precision_weight. Precision weight defaults to 0.
        :param mcc_weight: float
            Set the weight for the MCC score. The value should be between 0 and 1. This will adjust the weight for the
            model score as (w_acc * accuracy) + (w_prec * precision) + (w_mcc * mcc), with the w_acc and w_prec
            adjusted from the mcc_weight.
        """
        super().__init__(x_train, x_test, y_train, y_test, features, labels, model, cv)
        self.param_dict = None
        self.precision_w = precision_weight
        self.mcc_w = mcc_weight
        self.seed = None

    def search_params(self, param_dict: Union[dict, str] = 'default', show_param: bool = False):
        """
        Set or view search parameters. Params should be given as a dictionary for the given model, where the key
        corresponds to the parameter and the value containing any combination of list or tuple.
        A list is categorical, a tuple (low, high) is an int/float range, with an optional third element "log" for
        log-scale sampling or a step size. A dict is a conditional sub-grid: its keys are the choices, and each value
        is the grid searched only when that choice is picked, e.g. {"solver": {"saga": {...}, "lbfgs": {}}}.
        :param param_dict: dict
            Parameters for a sklearn model.
        :param show_param: bool
            Will print out the search space parameters to be used for optimization.
        :return:
        """
        # pull model
        model = self.model
        pull_model = _pull_model(model)

        if param_dict == 'default':
            params = _get_test_params(model=pull_model)
            self.param_dict = params
        elif isinstance(param_dict, dict):
            self.param_dict = param_dict
        else:
            raise ValueError("Search Parameters for model must be given!")

        if show_param:
            print(f"Set Search Params: {self.param_dict}")

    def opt_model(self, n_trials: int, direction="maximize",
                  callbacks: Optional[Iterable[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]] = None,
                  timeout: int = None, progress: bool = True, n_jobs=-1, seed: int = 42, verbose=False):
        """
        Optimize the model using Optuna
        :param n_trials: int
            The number of trials to run.
        :param direction: str
            Set direction of optimization. Can be minimize or maximize.
        :param callbacks: Optional[Iterable[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]]
            Set callbacks during model optimization.
        :param progress: bool
            Show progress bar.
        :param timeout: int
            Add a timeout to stop the model. Time must be in seconds. If set to None, trial will run until trials finish.
        :param n_jobs: int
            Set the number of jobs to run in parallel. Defaults to -1 (i.e. all)
        :param seed: int
            Set for reproducibility
        :param verbose: bool
            If True, will print the param grid for each trial
        :return:
        """

        # set seed
        sampler = TPESampler(seed=seed)

        # set study
        study = optuna.create_study(direction=direction, sampler=sampler)

        # set trials
        study.optimize(lambda trial: self._trial_space(trial, verbose=verbose), n_jobs=n_jobs, n_trials=n_trials,
                       timeout=timeout, show_progress_bar=progress, callbacks=callbacks)

        return study

    def build_opt_model(self, model: Optional[Union[str, BaseEstimator]] = None, params: Union[dict, str] = None):
        """
        Create an optimized model from the best parameters.
        :param model: Optional[Union[str, BaseEstimator]]
            Optional. Declare which model to use. Can be a string name, string name in short hand, or as a
            BaseEstimator.
        :param params: Union[dict, str]
            The best parameters obtained from optimization. Must be in dict format.
        :return:
        """
        # check model instance variable
        if model is None:
            model = self.model
        elif isinstance(self.model, BaseEstimator):
            model = self.model
        elif isinstance(self.model, str):
            model = _pull_model(self.model)
            self.model = model  # set model to class BaseEstimator
        # if input is given
        elif isinstance(model, str):
            model = _pull_model(model)
        elif isinstance(model, BaseEstimator):
            model = model
        else:
            raise ValueError("Model must be given when declaring Sklearn_opt!")

        # set params
        if not isinstance(params, (dict, str)):
            raise ValueError("Parameters must be given as a dictionary or a string to a json file!")
        elif isinstance(params, str):
            with open(params, 'r') as f:
                params = json.load(f)
        elif params is None:
            raise ValueError("Parameters cannot be None!")
        else:
            pass

        # set params
        model = model.set_params(**params)

        return model

    @staticmethod
    def best_solutions(space: optuna.study, savepath: Optional[str] = None):
        """
        Save solutions as txt file.
        :param space: optuna.study
            The search space obtained from opt_model.
        :param savepath: str
            Save path for best solutions. Will be saved as a json file.
        :return: dict of best solutions
        """
        params = space.best_params

        if savepath:
            import json
            with open(savepath, "w") as f:
                json.dump(params, f, indent=4)

        return params

    def _trial_space(self, trial, verbose=False):
        """Function to build and score model"""
        # pull model
        model = self.model
        pull_model = _pull_model(model)

        # pull default params if params not given
        if self.param_dict is None:
            self.param_dict = _get_test_params(model=pull_model)
            param_grid = self.param_dict
        else:
            param_grid = self.param_dict
        # # for troubleshooting
        # print(f"No search parameters given. Using default for {pull_model}:")

        # split params grid for trials
        trial_params = _parse_params(trial, param_grid, pull_model)

        if verbose:
            print("Trial_params:", trial_params)
            print("pulled model", pull_model)

        # set model
        if isinstance(pull_model, BaseEstimator):  # if model is instantiated but not fitted
            trial_model = pull_model.set_params(**trial_params)
        else:
            trial_model = pull_model(**trial_params)

        score = self._opt_score(trial_model, cv=self.cv)

        return score

    def _opt_score(self, trial_model, cv):
        """support function to get the score"""
        if cv is None:
            try:
                trial_model.fit(self.x_train, self.y_train)
            except (ValueError, TypeError) as e:
                print(e, " Pruning...")
                raise optuna.exceptions.TrialPruned()

            # get scores
            predictions = trial_model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions)
            mcc = matthews_corrcoef(self.y_test, predictions)

            # set weights
            t_weight = 1.0 - self.mcc_w
            w_prec = t_weight * self.precision_w
            w_acc = t_weight - w_prec

            # get fitness score
            opt_score = (accuracy * w_acc) + (precision * w_prec) + (mcc * self.mcc_w)

            return opt_score
        # get cv scores
        else:
            cross_val = CrossVal()
            try:
                results = cross_val.get_cv_scores(self.features, self.labels, model=trial_model, cv=self.cv)
            except (ValueError, TypeError) as e:
                print(e, " Pruning...")
                raise optuna.exceptions.TrialPruned()

            # set weights
            t_weight = 1.0 - self.mcc_w
            w_prec = t_weight * self.precision_w
            w_acc = t_weight - w_prec

            mcc = results['MCC_avg']
            accuracy = results['Accuracy_avg']
            precision = results['Precision_avg']

            # get fitness score
            opt_score = (accuracy * w_acc) + (precision * w_prec) + (mcc * self.mcc_w)

            return opt_score


"""Support Functions"""


def _get_test_params(model=None):
    """
    Support function to get parameters for skmodels.
    """

    if isinstance(model, type):
        model_name = model.__name__.lower()  # get the class name as a string
    elif isinstance(model, object):
        model_name = model.__class__.__name__.lower()  # get the class name of the instance
    else:
        raise ValueError(f"Model should be a class or an instance. Got {type(model)}")

    # Pull param grid from PARAM_LIST by comparing model names
    for param in PARAM_LIST:
        if model_name == param.name:
            return param.param_grid
        elif model_name == param.estimator.__name__.lower():  # Compare class name of estimator
            return param.param_grid
        elif isinstance(model_name, XGBClassifier) and isinstance(param.estimator, XGBClassifier):  # for XGBClassifier
            return param.param_grid
    raise ValueError(f"Params for {model_name} not found! Try setting it using Sklearn_opt.search_params()")


def _parse_params(trial, param_grid, model):
    """
    support function to parse the sklearn parameters into formats for int, float, or categorical.
    """
    trial_param = {}

    for param, value in param_grid.items():
        # skip parameters that aren't part of the model's hyperparameters
        if param not in param_grid:
            continue

        # conditional sub-grid: pick a key, then search only the params valid for that choice
        if isinstance(value, dict):
            choice = trial.suggest_categorical(param, list(value))
            trial_param[param] = choice
            trial_param.update(_parse_params(trial, value[choice], model))
            continue

        # fixed values
        if not isinstance(value, (tuple, list)):
            trial_param[param] = value
            continue

        # trial params for int, float, or categorical
        if isinstance(value, tuple) and len(value) in (2, 3):
            low, high = value[:2]
            # optional third element: "log" for log-scale sampling, otherwise a step size
            extra = value[2] if len(value) == 3 else None
            kwargs = {} if extra is None else {"log": True} if extra == "log" else {"step": extra}
            if isinstance(low, int) and isinstance(high, int):
                trial_param[param] = trial.suggest_int(param, low, high, **kwargs)
            else:
                trial_param[param] = trial.suggest_float(param, low, high, **kwargs)
        elif isinstance(value, list):
            trial_param[param] = trial.suggest_categorical(param, value)

    # print("trial parameters", trial_param) # for troubleshooting
    return trial_param


def _pull_model(query_model=None):
    """
    Support function to pull skmodel.
    """
    # pull param grid from NamedTuple list if string
    for model in PARAM_LIST:
        if query_model == model.name or query_model == model.full_name:
            return model.estimator()
        else:
            pass

    # if input is a model that is instance but not fitted
    if isinstance(query_model, BaseEstimator):
        return query_model
    # if pulling from model not in constant
    elif isinstance(query_model, str):
        skmodel = Models()
        pull_model = skmodel.get_sk_model(query_model)
        return pull_model
    else:
        raise ValueError(f"Cannot find {query_model} . Try 'lr', 'rf', 'svc', 'knn', 'xgb'.")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
