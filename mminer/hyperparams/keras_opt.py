"""
Script to run hyperparameter optimizations.
"""
import json
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Optional, Union, Iterable, Callable
from mminer.models.split import CrossVal
from mminer.hyperparams.optimization import Optimization

# Set keras backed dependencies and MacOS MPS Fallback to CPU if needed
import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras.src.metrics import Recall
from keras.src.regularizers import L1, L2, L1L2
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Input
from keras._tf_keras.keras.optimizers import *
from keras._tf_keras.keras.metrics import Precision
from keras._tf_keras.keras.callbacks import EarlyStopping

# For XGBoost reference
# https://www.kaggle.com/code/mh0386/facial-emotions-detection#SVM

# set fallback
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


class Keras_opt(Optimization):
    def __init__(self,
                 x_train: Optional[Union[np.array, pd.DataFrame]] = None,
                 x_test: Optional[Union[np.array, pd.DataFrame]] = None,
                 y_train: Optional[Union[np.array, pd.DataFrame]] = None,
                 y_test: Optional[Union[np.array, pd.DataFrame]] = None,
                 features: Optional[Union[np.array, pd.DataFrame]] = None,
                 labels: Optional[Union[np.array, pd.DataFrame]] = None,
                 feat_cols: Optional[Union[list, str]] = None,
                 activity_cols: Optional[Union[list, str]] = None,
                 model: Optional[Union[keras.Model]] = None,
                 cv: Optional[int] = None,
                 precision_weight: float = 0,
                 loss_weight: float = 1.0
                 ):
        """
        Initialize KerasGA. These inputs are optional, however, giving the parameters when initializing the class
        should make the optimization process easier.
        :param x_train: Optional[Union[np.array, pd.DataFrame]]
            Input the training features for the given problem.
        :param x_test: Optional[Union[np.array, pd.DataFrame]]
            Input the testing features for the given problem.
        :param y_train: Optional[Union[np.array, pd.DataFrame]]
            Input the training labels for the given problem.
        :param y_test: Optional[Union[np.array, pd.DataFrame]]
            Input the testing labels for the given problem.
        :param features: Optional[Union[np.array, pd.DataFrame]]
            Input the features for the entire dataset.
        :param labels: Optional[Union[np.array, pd.DataFrame]]
            Input the labels for the entire dataset.
        :param feat_cols: Optional[Union[list, str]]
            Set the data corresponding to the feature columns.
        :param activity_cols: Optional[Union[list, str]]
            Set the data corresponding to the activity columns.
        :param model: Optional[Union[keras.Model]]
            Set model to optimize. If none given, models can be called using the class methods.
        :param cv: Optional[int]
            Set the number of cv folds. found with the dataset. The features/labels/train/test sets should have the
            corresponding cv folds labeled.
        :param precision_weight: float
            Set the weight for precision score. The value should be between 0 and 1. The corresponding weight for
            accuracy will be calculated as 1 - precision_weight, adjusted to the weight of the loss. Precision weight
            defaults to 0.
        :param loss_weight: float
            Set the weight for loss score. The value should be between 0 and 1. This will adjust the weight for the
            model score as (w_acc * val_acc) + (w_prec * val_prec) + (w_loss * n_loss), with the w_acc and w_prec
            adjusted from the loss_weight.
        """
        super().__init__(x_train, x_test, y_train, y_test, features, labels, model, cv)
        self.feat_cols = feat_cols
        self.activity_cols = activity_cols
        self.precision_w = precision_weight
        self.loss_w = loss_weight

    def opt_model(
            self,  # Define the search space
            n_trials: int = 10,
            direction: str = "maximize",
            patience: int = 30,
            max_layer: int = 3,
            units: tuple = (10, 100),
            unit_decay: tuple = (0, 0.5),
            batch_size: tuple = (10, 50),
            epochs: tuple[int, int] = (25, 50),
            activation: list = ["relu", "tanh", "sigmoid", "selu", "elu"],
            optimizers: list = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Nadam", "Ftrl"],
            learning_rate: tuple = (0.00001, 0.01),
            regularizer: list = ["l1", "l2", "l1l2"],
            max_dropout: float = 0.5,
            progress: bool = True,
            callbacks: Optional[Iterable[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]]] = None,
            timeout: int = None,
            n_jobs: int = -1,
            seed: int = 42,
    ):
        """
        Generate an optimized keras model using the GA algorithm.
        :param n_trials: int
            Set the number of trials during optimization.
        :param direction: str
            Set the direction of the optimization. Can be "maximize" or "minimize".
        :param patience: int
            Set the number of patience for early stopping. This param is not optimized and should be set by the user.
            Defaults to 30.
        :param max_layer: int
            Determines the maximum number of layers to generate for the Keras model.
        :param units: tuple
            Define the size of the dense layer neurons.
        :param unit_decay: tuple
            Set the option to reduce the number of units for each layer.
        :param batch_size: tuple
            Define the size of the subset of training dataset before backpropagation occurs.
        :param epochs: int
            An epoch is one pass of the entire training dataset. Otherwise, known as how many times the model goes
            through the training dataset.
        :param activation: list
            Activation functions to transform the neuron input values. None will default to ["relu", "sigmoid", "tanh"].
        :param optimizers: dict
            A list of Keras optimizers. If None, will default to all available optimizers.
        :param learning_rate: tuple
            Set the range for the learning rate of optimizers
        :param regularizer: list
            Set the different regularizers for use. This will be used for kernal_regularizer only.
        :param max_dropout: float
            Set the dropout rate. This is used to help prevent overfitting.
        :param progress: bool
            Set progress bar for optimization.
        :param callbacks:
            Set the callbacks during model optimization.
        :param timeout: int
            Add a timeout to stop the model. Time must be in seconds. If set to None, trial will run until trials finish.
        :param n_jobs: Union[list,int]
            Select number of threads to use for parallelization. Must be an integer >= 1.
        :param seed: int
            Set random seed for reproducibility.
        :return:
        """
        # set seed
        sampler = TPESampler(seed=seed)

        # set study
        study = optuna.create_study(direction=direction, sampler=sampler)

        study.optimize(
            lambda trial: self._trial_space(trial, max_layer, units, unit_decay, batch_size, max_dropout, activation,
                                            optimizers, epochs, learning_rate, regularizer, patience), n_jobs=n_jobs,
            n_trials=n_trials, timeout=timeout, show_progress_bar=progress, callbacks=callbacks)

        return study

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

    def build_opt_model(self, params: Union[dict, str] = None, metrics: Union[str, list] = "mini", patience: int = 30):
        """
        Create an optimized model from the best parameters.
        :param params: Union[dict, str]
            The best parameters obtained from keras optimization. Must be in dict format. The Keras model will then be
            created from the input dictionary params.
        :param metrics: Union[str, list]
            Set the type of metrics to output of the model. Can be 'mini' or 'all' is supported, and will produce
            ['accuracy', 'precision'] or ['accuracy', Precision(), Recall()], respectively. If a list is given, items
            must correspond to Keras metric classes.
        :param patience: int
            Set the model early stopping. Defaults to 30.
        :return:
        """

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

        # set model shape
        shape = (self.features.shape[1],)

        # set input dictionary into settings, else set to None
        activation = params.get("activation", None)
        dropout = params.get("dropout", None)
        l1_lambda = params.get("l1_lambda", None)
        l2_lambda = params.get("l2_lambda", None)
        learning_rate = params.get("lr", None)
        momentum = params.get("momentum", None)
        n_layers = params.get("n_layers", None)
        optimizer = params.get("optimizer", None)
        reg_type = params.get("reg_type", None)
        units = params.get("units", None)
        unit_decay = params.get("unit_decay", None)

        # build model
        trial_model, trial_callback = self._build_model(shape, activation, dropout, l1_lambda, l2_lambda, learning_rate,
                                                        momentum, n_layers, optimizer, reg_type, units, unit_decay,
                                                        patience, metrics=metrics)

        return trial_model

    def _trial_space(self, trial, max_layer, units, unit_decay, batch_size, max_dropout, activations, optimizers,
                     epochs, learning_rate, regularizer, patience):
        # hyperparams for tuning
        global trial_opt
        n_layers = trial.suggest_int("n_layers", 1, max_layer)
        units = trial.suggest_int("units", units[0], units[1])
        batch_size = trial.suggest_int("batch_size", batch_size[0], batch_size[1])
        dropout = trial.suggest_float("dropout", 0, max_dropout)
        activation = trial.suggest_categorical("activation", activations)
        epochs = trial.suggest_int("epochs", epochs[0], epochs[1])

        # determine reduction of unit decay for each layer
        unit_decay = trial.suggest_float("unit_decay", unit_decay[0], unit_decay[1])

        # optimizers
        optimizer = trial.suggest_categorical("optimizer", optimizers)
        # optimizer hyperparameters
        learning_rate = trial.suggest_float("lr", learning_rate[0], learning_rate[1], log=True)
        momentum = trial.suggest_float("momentum", 0.0, 0.9) if optimizer == "sgd" else 0.0

        # set regularizers
        reg_type = trial.suggest_categorical("regularizer", regularizer)
        # set regularizer strength
        l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-2, log=True)
        l2_lambda = trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True)

        # build, fit, and obtain model scores
        score = self._opt_score(activation, dropout, l1_lambda, l2_lambda, learning_rate, momentum, n_layers,
                                optimizer, reg_type, units, unit_decay, batch_size, epochs, patience, cv=self.cv)

        return score

    def _opt_score(self, activation, dropout, l1_lambda, l2_lambda, learning_rate, momentum, n_layers, optimizer,
                   reg_type, units, unit_decay, batch_size, epochs, patience, cv):
        """support function to fit the model and get the score"""
        if cv is None:
            try:
                # build model
                shape = (self.features.shape[1],)
                trial_model, trial_callback = self._build_model(shape, activation, dropout, l1_lambda, l2_lambda,
                                                                learning_rate, momentum, n_layers, optimizer, reg_type,
                                                                units, unit_decay, patience)

                # fit model
                trial_history = trial_model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                                                validation_data=(self.x_test, self.y_test), callbacks=[trial_callback])
            except (ValueError, TypeError) as e:
                print(e, " Pruning...")
                raise optuna.exceptions.TrialPruned()

            # get scores
            val_loss = np.mean(trial_history.history["val_loss"])
            val_acc = np.mean(trial_history.history["val_accuracy"])
            val_prec = np.mean(trial_history.history["val_precision"])

            # convert loss from 0 to 1 scale
            n_loss = 1 / (1 + val_loss)

            # ensure weight input ranges from 0 to 1
            assert 0.0 <= self.loss_w <= 1.0
            assert 0.0 <= self.precision_w <= 1.0

            # set weights, must total 1
            t_weight = 1.0 - w_loss
            w_prec = t_weight * self.precision_w
            w_acc = t_weight - w_prec

            # Final score (range 0 to 1)
            score = (w_acc * val_acc) + (w_prec * val_prec) + (self.loss_w * n_loss)

            return score
        # get cv scores
        else:
            # ensure the input table with the CV labels are removed. Used to get shape of model.
            cross_val = CrossVal()
            features_cv = cross_val.slice_data(data=self.features, cols=self.feat_cols)
            labels_cv = cross_val.slice_data(data=self.labels, cols=self.activity_cols)

            # build model
            shape = (features_cv.shape[1],)
            trial_model, trial_callback = self._build_model(shape, activation, dropout, l1_lambda, l2_lambda,
                                                            learning_rate, momentum, n_layers, optimizer, reg_type,
                                                            units, unit_decay, patience)

            # get cv scores
            cross_val = CrossVal()
            scores = cross_val.get_cv_scores(self.features, self.labels, model=trial_model, cv=self.cv,
                                             callbacks=trial_callback, epochs=epochs, batch_size=batch_size)

            # get scores
            val_loss = np.mean(scores.get("Loss"))
            val_acc = np.mean(scores.get("Accuracy"))
            val_prec = np.mean(scores.get("Precision"))

            # convert loss from 0 to 1 scale
            n_loss = 1 / (1 + val_loss)

            # ensure weight input ranges from 0 to 1
            assert 0.0 <= self.loss_w <= 1.0
            assert 0.0 <= self.precision_w <= 1.0

            # set weights, must total 1
            t_weight = 1.0 - self.loss_w
            w_prec = t_weight * self.precision_w
            w_acc = t_weight - w_prec

            # Final score (range 0 to 1)
            score = (w_acc * val_acc) + (w_prec * val_prec) + (self.loss_w * n_loss)

            return score

    def _build_model(self, shape, activation, dropout, l1_lambda, l2_lambda, learning_rate,
                     momentum, n_layers, optimizer, reg_type, units, unit_decay, patience,
                     metrics: Union[str, list] = "mini"):
        """support function to build the model from trials"""
        global trial_opt
        # build model
        model = Sequential()
        model.add(Input(shape=shape))
        # add layers
        for x in range(n_layers):
            # decay the number of units for each layer
            if unit_decay != 0.0:
                layer_units = max(2, int(units * (unit_decay ** x)))
            else:
                layer_units = units

            if reg_type == "l1":
                model.add(Dense(units=layer_units, activation=activation, kernel_regularizer=L1(l1_lambda)))
            elif reg_type == "l2":
                model.add(Dense(units=layer_units, activation=activation, kernel_regularizer=L2(l2_lambda)))
            else:  # L1L2
                model.add(
                    Dense(units=layer_units, activation=activation, kernel_regularizer=L1L2(l1_lambda, l2_lambda)))

            # dropout layer
            model.add(Dropout(rate=dropout))

        # output layer
        model.add(Dense(1, activation="sigmoid"))

        # select optimizer for trial
        optimizer = optimizer.lower()  # try to normalize optimizer cases
        if optimizer == "adam":
            trial_opt = Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            trial_opt = SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == "rmsprop":
            trial_opt = RMSprop(learning_rate=learning_rate)
        elif optimizer == "adagrad":
            trial_opt = Adagrad(learning_rate=learning_rate)
        elif optimizer == "nadam":
            trial_opt = Nadam(learning_rate=learning_rate)
        elif optimizer == "adadelta":
            trial_opt = Adadelta(learning_rate=learning_rate)
        elif optimizer == "ftrl":
            trial_opt = Ftrl(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # set fallback
        model.to(DEFAULT_DEVICE)

        # options for different metric options
        metric_options = {
            "mini": ["accuracy", Precision(name="precision")],
            "all": ["accuracy", Precision(name="precision"), Recall(name="recall")],
        }

        # defaults to "mini"
        if isinstance(metrics, str):
            compile_metrics = metric_options.get(metrics, None)
            if compile_metrics is None:
                raise ValueError('Only "mini" or "all" are valid metrics.')
        elif isinstance(metrics, list):
            compile_metrics = metrics
        else:
            raise ValueError("Model Metrics must be a list of Keras metrics or a string such as 'mini' or 'all'!")

        # compile model
        model.compile(optimizer=trial_opt, loss="binary_crossentropy",
                      metrics=compile_metrics)

        # early stopping
        callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=1, mode="auto")

        return model, callback


if __name__ == "__main__":
    import doctest

    doctest.testmod()
