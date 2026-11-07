"""
Script super class for hyperparameter optimizations.
"""

import os
import json
import math
import pandas as pd
import numpy as np
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple
from sklearn.base import BaseEstimator

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras._tf_keras.keras.utils import plot_model

"""
The following scripts are for generating plots using optuna optimization
"""


class Optimization:
    def __init__(self,
                 x_train: Union[np.array, pd.DataFrame] = None,
                 x_test: Union[np.array, pd.DataFrame] = None,
                 y_train: Union[np.array, pd.DataFrame] = None,
                 y_test: Union[np.array, pd.DataFrame] = None,
                 features: Union[np.array, pd.DataFrame] = None,
                 labels: Union[np.array, pd.DataFrame] = None,
                 model: Union[BaseEstimator, keras.Model] = None,
                 cv: Optional = None,
                 ):
        """
        Initialize genetic optimizations. These inputs are optional, however, giving the parameters when initializing
        the class should make the optimization process easier.
        :param model: Keras model
            Optional. This parameter should hold a Keras model. If None is given, then a default Keras model will be
            generated instead. It can be viewed using the model.summary() to visualize model layout.
        :param features: np.array or pd.DataFrame
            Input the features for the given problem.
        :param labels: np.array or pd.DataFrame
            Input labels for the given problem.
        :param cv: A list of tuples
            Input a list of tuples containing indexes for the cross validation. This can be obtained using the
            CrossVal() class.
        """
        self.model = model
        self.features = features
        self.labels = labels
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv

    """load params from txt"""

    def load_params(self, path: str = None):
        # load a json file containing model parameters
        with open(path, 'r') as f:
            params = json.load(f)

        return params

    """plot methods"""

    def plot_trials(
            self,
            space: optuna.study = None,
            title: str = "Trials vs. Optimization Score",
            xlabel: str = 'Number of Trials',
            ylabel: str = 'Score',
            color: str = "#1f77b4",
            linewidth: int = 2,
            fontsize: int = 14,
            figsize: Tuple[int, int] = (10, 6),
            savepath: Optional[str] = None,
    ):
        """
        Plot a line plot of trials vs optimization score.
        :param space: optuna.study
            The optimization results from opt_model.
        :param title:   str
            Set the title of the figure.
        :param xlabel: str
            Set the title of the x-axis
        :param ylabel: str
            Set the title of the y-axis
        :param color: str
            Color of the line. Can be name of color or a hex code.
        :param linewidth: int
            Set the width of the line.
        :param fontsize: int
            Set the fontsize of the figure.
        :param figsize: Tuple[int, int]
            Set the figure size.
        :param savepath: str
            Set figure save path.
        :return:
        """
        # results from opt_model
        space_df = space.trials_dataframe()

        plt.figure(figsize=figsize)
        sns.lineplot(data=space_df, x='number', y='value', color=color, linewidth=linewidth)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def plot_search_space(self, study: optuna.study.Study, title: str = None, title_size: int = 20, fontsize: int = 14,
                          height: int = 2, marker_size: int = 25, color: list = None, savepath: Optional[str] = None):
        """
        Plot a graph describing the search space using Optuna results.
        :param study: optuna.study.Study
            The study object containing the optimization results.
        :param title: str, optional
            The title of the plot.
        :param title_size: int, optional
            The font size for the title.
        :param fontsize: int, optional
            The font size for the axis labels and tick labels.
        :param height: int, optional
            The height of the individual plots.
        :param marker_size: int, optional
            The marker size for the scatter plot.
        :param color: list, optional
            Colors for different plot elements (e.g., ['Greens', 'red', 'blue']).
        :param savepath: str, optional
            The path to save the figure.
        """
        # results from opt_model
        trials_df = study.trials_dataframe()

        # process and extract the hyperparameter columns
        df_params = trials_df.filter(regex='params_', axis=1)
        cols = df_params.columns.str.replace('params_', '', regex=True)
        df_params = df_params.rename(columns=dict(zip(df_params.columns, cols)))
        # remove columns n_jobs and random_state associated with sklearn models
        df_params = df_params.drop(columns=[col for col in ['n_jobs', 'random_state'] if col in df_params])

        # convert categorical data to numbers
        for col in df_params.columns:
            if df_params[col].dtype == 'object':
                # mapp category
                df_params[col] = df_params[col].astype('category')

                # print mapping
                mapping = dict(enumerate(df_params[col].cat.categories))
                print(f"Category mapping for {col}")
                for code, label in mapping.items():
                    print(f"  {code} → {label}")
                # Replace column with codes
                df_params[col] = df_params[col].cat.codes

        if title is None:
            title = "Search Space"

        sns.set_style("white")

        if color:
            color_kdeplot = color[0]
            color_dots = color[1]
            color_line = color[2]
        else:
            color_kdeplot = "ch:s=.25,rot=-.25"
            color_dots = 'red'
            color_line = 'red'

        # Generate plots
        img = sns.PairGrid(df_params, diag_sharey=False, height=height)
        img = img.map_upper(sns.scatterplot, s=marker_size, color=color_dots, alpha=0.2)
        img = img.map_lower(
            sns.kdeplot,
            fill=True,
            cmap=sns.color_palette(color_kdeplot, as_cmap=True),
        )
        img = img.map_diag(sns.kdeplot, fill=True, alpha=0.2, color=color_line, linewidth=1.5)

        # adjust labels
        for ax in img.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

            # adjust tick labels for each subplot
            ax.tick_params(axis='both', labelsize=12)

        plt.suptitle(title, y=1.02, fontsize=title_size)

        # Save figure if savepath is provided
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)

        plt.show()

    def plot_model(
            self,
            model: Optional[Union[keras.Model, keras.Sequential]] = None,
            show_fig: bool = False,
            savepath: str = 'model.png',
            show_shapes: bool = True,
            show_dtype: bool = True,
            show_layer_names: bool = False,
            direction: str = "TB",
            expand_nested: bool = False,
            dpi: int = 300,
            show_layer_activations: bool = True,
            show_trainable: bool = False,
            **kwargs
    ):
        """
        Converts a Keras model to dot format and save to a image file.

        :param model: keras.Model
            Input Keras model to convert.
        show_fig: bool
            Display the model figure. By default, the model will be saved as a .png figure, but will not be displayed if
            used in a Jupyter notebook. Set to True to view image.
        :param savepath: str
            File path to save image.
        :param show_shapes: bool
            To display shape information.
        :param show_dtype: bool
            To display layer dtypes.
        :param show_layer_names: bool
            To display layer names.
        :param direction: str
            A string to specify the format of the plot. "TB" creates a vertical plot. "LR" creates a horizontal plot.
        :param expand_nested: bool
            To expand the nested Functional models into clusters.
        :param dpi: int
            Image resolution in dots per inch.
        :param show_layer_activations: bool
            To display layer activations (Only for layers with activation property).
        :param show_trainable: bool
            To display if a layer is trainable.
        :return:
        """
        # set model to self.ga_instance
        if isinstance(model, keras.Model) or isinstance(model, keras.Sequential):
            pass
        else:
            raise AttributeError('Not a Keras model!')

        fig = plot_model(
            model=model,
            to_file=savepath,
            show_shapes=show_shapes,
            show_dtype=show_dtype,
            show_layer_names=show_layer_names,
            rankdir=direction,
            expand_nested=expand_nested,
            dpi=dpi,
            show_layer_activations=show_layer_activations,
            show_trainable=show_trainable,
            **kwargs)

        if show_fig is True:
            return fig


class EarlyStopping:
    """Set early stopping criteria. Only successful trials will be counted"""

    def __init__(self, patience: int = 30, n_prune: int = 30) -> None:
        self.patience = patience
        self.prune_threshold = n_prune
        self.n_prune = 0
        self.best_score = None
        self.unimproved_trials = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        # handle consecutive pruning:
        if trial.state == TrialState.PRUNED:
            self.n_prune += 1
        else:
            self.n_prune = 0

        # trigger statement
        if self.n_prune >= self.prune_threshold:
            print(f"Early stopping triggered: {self.n_prune} consecutive trials pruned!")
            study.stop()
            return

        # handle no score improvements
        if trial.state != TrialState.COMPLETE or trial.value is None or math.isnan(trial.value):
            return

        # valid score
        if self.best_score is None:
            self.best_score = trial.value
            return

        # check if current trial is better than best value
        if trial.value > self.best_score:
            self.best_score = trial.value
            self.unimproved_trials = 0
        else:
            self.unimproved_trials += 1

        # trigger statement
        if self.unimproved_trials >= self.patience:
            print(f"Early stopping triggered: No improvement after {self.unimproved_trials} trials!")
            study.stop()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
