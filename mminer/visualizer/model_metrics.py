"""
Script to generate model metrics or statistic comparisons.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Union, Optional
import re
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    roc_curve,
    PrecisionRecallDisplay,
    average_precision_score,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve, LearningCurveDisplay

__all__ = ["ModelMetrics", "ModelStats", "ConfusionMatrix", "ConfusionMatrixDisplay", "ClassificationReport",
           "plot_enrichment"]

"""
Scripts for comparing classification models.
"""


class ModelStats:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def flatten_array_data(self, score_col: str = None, model_col: str = "Model"):
        """
        Support function. This will accept a column containing an array. The column will be flattened and output a final
        dataframe containing the Model labels and its indicated scoring metric.

        :param score_col: str
            Name of column containing scores.
        :param model_col:
            Name of column containing model labels. Default is "Model"
        :return:
        """

        data = self.data.copy()

        # slice DataFrame
        data = data[[model_col, score_col]]

        # Flatten array by input column and convert to numeric
        flatten_df = data.explode(column=score_col).reset_index(drop=True)
        flatten_df[score_col] = pd.to_numeric(flatten_df[score_col], errors="coerce")

        return flatten_df


"""
Scripts for comparing classification models.
"""


class ModelMetrics:
    def __init__(self, X_train=None, X_test=None, y_train=None, y_test=None, feat_col=None, label_col=None):
        """
        Params are used to initialize the ModelMetrics class. If only 'feat_col' and 'label_col' are given, then they
        will be used as X_test and y_test respectively.
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param feat_col:
        :param label_col:
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if X_train is None and X_test is None and y_train is None and y_test is None:
            self.X_test = feat_col
            self.y_test = label_col

    def classification_report(self, model, fit_model: bool = False, labels: list = None, support: bool = True,
                              cmap: str = "Blues", cbar: bool = True, plot: bool = True, title: Optional[str] = None,
                              savepath: str = None):
        """
        Classification report that shows the precision, recall, F1, and support scores for the model. Integrates
        numerical scores as well as a color-coded heatmap.

        :param model: str
            Name of classification model. Must be from SKlearn
        :param labels: list of str
            A list of strings that correspond to the class labels of the classification model.
        :param support: bool
            The number of actual occurrences of the class in the indicated dataset.
        :param cmap: str
            Specify a colormap to define the heatmap.
        :param cbar: bool
            Whether to set the colorbar
        :param plot: bool
            Function will output a plot of the YellowBrick classification report by default. If running multiple models,
            it may be better to print the report in the terminal. This is done if plot is set to False.
        :param savepath: str
            Save path for the Classification report plot.
        :return:
        """

        if labels is None:
            labels = ["Inactive", "Active"]

        # fit model if needed
        if not fit_model:
            model.fit(self.X_train, self.y_train)

        # generate predictions and score
        y_pred = model.predict(self.X_test)

        # plot
        if plot:
            ax = ClassificationReport(self.y_test, y_pred, support=support, class_names=labels, cbar=cbar, cmap=cmap)

            # extract model type to add to plot
            if title is None:
                raw_name = model.__class__.__name__
                clean_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_name)
                title = f"{clean_name} Classification Report"

            plt.title(title)
            plt.tight_layout()

            # savepath
            if savepath:
                plt.savefig(savepath, dpi=300)
            plt.show()

        elif not plot:
            report = classification_report(self.y_test, y_pred, target_names=labels, digits=3)
            return report
        return None

    def confusion_matrix(self, model, fit_model: bool = False, labels: Optional[Union[list, str]] = None,
                         cmap: str = "Blues", plot: bool = True, title: Optional[str] = None, cbar: bool = True,
                         savepath: str = None):
        """
        Confusion matrix for the model.

        :param model: str
            A SKlearn class model.
        :param fit_model: bool
             If model is already fit, then the confusion matrix will be generated from fit data. Otherwise, model will
             be fit using the parmas given on ModelMetrics() instance.
        :param labels: Optional[Union[list, str]
            A list of strings that correspond to the class labels of the classification model.
        :param cmap: str
            Specify a colormap to define the heatmap.
        :param plot: bool
            Function will output a plot of the YellowBrick classification report by default. If running multiple models,
            it may be better to print the report in the terminal. This is done if plot is set to False.
        :param title: Optional[str]
            Set the title for the confusion matrix plot.
        :param cbar: bool
            Plot a cbar with the matrix. This is only available when percent is set to 'all'.
        :param savepath: str
            Save path for the Classification report plot.
        :return:
        """

        if labels is None:
            labels = ["Inactive", "Active"]

        # fit model if needed
        if not fit_model:
            model.fit(self.X_train, self.y_train)

        # generate predictions and score
        y_pred = model.predict(self.X_test)

        # plot
        if plot:
            ax = ConfusionMatrix(self.y_test, y_pred, labels, cbar, cmap)

            # savepath
            plt.tight_layout()

            if savepath and title:
                plt.title(title)
                plt.tight_layout()
                plt.savefig(savepath, dpi=300)
            elif savepath and title is None:
                plt.tight_layout()
                plt.savefig(savepath, dpi=300)
            elif savepath is None and title:
                plt.title(title)
                plt.tight_layout()
            plt.show()
        return None

    def predict_metrics(self, model, X_test=None, y_test=None):
        """
        Generate scoring metrics for an indicated sklearn model and predictions.
        :param model:
            Query sklearn model for predictions. Model must be already fitted.
        :param X_test: Data containing features to predict. This is optional and if given, it will overwrite the
        self.X_test.
        :param y_test: Data containing labels to predict. This is optional and if given, it will overwrite the
        self.y_test.
        :return: pd.DataFrame
        """

        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test

        model_predict = model.predict(X_test)
        accuracy = accuracy_score(y_test, model_predict)
        recall = recall_score(y_test, model_predict)
        precision = precision_score(y_test, model_predict)
        mcc = matthews_corrcoef(y_test, model_predict)

        # roc_auc_score requires true binary labels and probability score, not predicted class
        y_predict_probability = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_predict_probability)

        # place results in array and generate DataFrame
        model_name = model.__class__.__name__
        scores = [[model_name, accuracy, recall, precision, mcc, roc]]
        col_headers = ["Model", "Accuracy", "Recall", "Precision", "MCC", "ROC"]
        df = pd.DataFrame(scores, columns=col_headers)

        return df

    def plot_learning_curve(self, model, x_features: Union[pd.DataFrame, np.ndarray] = None,
                            y_values: Union[pd.DataFrame, np.ndarray] = None, score_name: str = "Accuracy",
                            title: str = "Learning Curve", figsize: tuple = (6.4, 4.8), savepath=None):
        """
        Genereate a learning curve plot for a sklearn model. Needs the data used for training and activity labels.
        :param model:
            An sklearn model. Must be already fitted.
        :param x_features: Union[pd.DataFrame, np.array]
            Input features used for training the data.
        :param y_values: Union[pd.DataFrame, np.array]
            Input labels used for training the data.
        :param score_name: str
            Label for the Y-axis. Defaults to Accuracy.
        :param title: str
            Set the title of the plot.
        :param figsize: tuple
            Set the figure size.
        :param savepath: str
            Indicate save location for the plot.
        :return:
        """
        # check instance variables
        if x_features is None:
            x_features = self.X_test
        if y_values is None:
            y_values = self.y_test

        # get learning_curve values
        train_sizes, train_scores, test_scores = learning_curve(model, x_features, y_values)

        # ensure default sklearn colors are used
        plt.rcdefaults()
        fig, ax = plt.subplots(figsize=figsize)

        # plot LearningCurveDisplay
        plot = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores,
                                    score_name=score_name)
        plot.plot(ax=ax)

        ax.set_title(title)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=300)

        plt.show()

    def plot_ROC(self, model: Union[list, BaseEstimator, object] = None, X_test=None, y_test=None,
                 title: str = None, label: Union[str, list] = None, color: Union[str, list] = None,
                 legend_loc: str = "lower right", figsize: tuple[float, float] = (10, 5), fontsize: int = 12,
                 titlesize: int = 15, grid: bool = False, savepath: str = None, **kwargs):
        """
        Plot ROC Curve. Can be for a single model or a list of models.
        :param model: [list, BaseEstimator, object]
            An sklearn or XGBoost model to plot. Can be a single model or a list of models.
        :param X_test:
            Dataset containing features to predict
        :param y_test:
            Dataset containing labels to predict
        :param title: str
            Title of the plot. If None is given, the default title will be used.
        :param label: Union[str, list]
            The labels for indicated model. If None is given, will default to ModelX, where X is a number.
        :param color: Union[str, list]
            Set color scheme for groups in plot. Can be a string or a list of colors.
        :param legend_loc: str
            Set the location of the figure legend.
        :param figsize: tuple
            Set figure size.
        :param fontsize: int
            Font size for axis
        :param titlesize: int
            Font size for title
        :param grid: bool
            Set grid.
        :param savepath: str
            Filepath to save figure.
        :param kwargs:
            A set of kwargs associated with sklearn RocCurveDisplay
        :return:
        """
        if X_test is None: X_test = self.X_test
        if y_test is None: y_test = self.y_test

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # set model, colors sand label conditions
        models = model if isinstance(model, list) else [model]
        colors = color if isinstance(color, list) else ([color] if color else [])
        labels = label if isinstance(label, list) else ([label] if label is not None else [])

        # loop through models list
        for i, mod in enumerate(models):
            current_color = colors[i] if i < len(colors) else None
            base_label = labels[i] if i < len(labels) else f"Model {i + 1}"

            if hasattr(mod, "predict_proba"):
                # ROC Curve
                RocCurveDisplay.from_estimator(mod, X_test, y_test, ax=ax, color=current_color, **kwargs)
                # calculate score
                # get proba
                y_prob = mod.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_prob)
                # update label
                ax.get_lines()[-1].set_label(f"{base_label} (AUC = {auc_score:.3f})")
            else:
                # keras model
                y_score = mod.predict(X_test)
                if y_score.shape[1] > 1:
                    y_score = y_score[:, 1]

                fpr, tpr, _ = roc_curve(y_test, y_score)
                auc_score = roc_auc_score(y_test, y_score)
                ax.plot(fpr, tpr, color=current_color, label=f"{base_label} (AUC = {auc_score:.3f})")

        # diagonal line for random chance (AUC = 0.5)
        ax.plot([0, 1], [0, 1], color='gainsboro', linestyle='--', label='Chance')

        if title is None: title = "ROC Curves"
        plt.grid(grid)
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=fontsize)
        plt.ylabel("True Positive Rate", fontsize=fontsize)
        plt.title(title, fontsize=titlesize)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=legend_loc)
        plt.tight_layout()
        if savepath: plt.savefig(savepath, dpi=300)
        plt.show()

    def plot_PR(self, model: Union[list, object] = None, X_test=None, y_test=None,
                title: str = None, label: Union[str, list] = None, color: Union[str, list] = None,
                legend_loc: str = "lower right", figsize: tuple[float, float] = (10, 5),
                fontsize: int = 12, titlesize: int = 15, grid: bool = False, savepath: str = None, **kwargs):
        """
        Plot Precision-Recall Curve. Can be for a single model or a list of models.
        :param model: Union[list, BaseEstimator]
           An sklearn or XGBoost model to plot. Can be a single model or a list of models.
        :param X_test:
            Dataset containing features to predict
        :param y_test:
            Dataset containing labels to predict
        :param title: str
            Title of the plot. If None is given, the default title will be used.
        :param label: Union[str, list]
            The labels for indicated model. If None is given, will default to ModelX, where X is a number.
        :param color: Union[str, list]
            Set color scheme for groups in plot. Can be a string or a list of colors.
        :param legend_loc: str
            Set the location of the figure legend.
        :param figsize: tuple
            Set figure size.
        :param fontsize: int
            Font size for axis
        :param titlesize: int
            Font size for title
        :param grid: bool
            Set grid.
        :param savepath: str
            Filepath to save figure.
        :param kwargs:
            A set of kwargs associated with sklearn RocCurveDisplay
       :return:
        :return:
        """
        if X_test is None: X_test = self.X_test
        if y_test is None: y_test = self.y_test

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # set model, colors sand label conditions
        models = model if isinstance(model, list) else [model]
        colors = color if isinstance(color, list) else ([color] if color else [])
        labels = label if isinstance(label, list) else ([label] if label is not None else [])

        # loop through models list
        for i, mod in enumerate(models):
            current_color = colors[i] if i < len(colors) else None
            base_label = labels[i] if i < len(labels) else f"Model {i + 1}"

            if hasattr(mod, "predict_proba"):
                # PrecisionRecall
                PrecisionRecallDisplay.from_estimator(mod, X_test, y_test, ax=ax, color=current_color, **kwargs)
                # calculate score
                # get proba
                y_prob = mod.predict_proba(X_test)[:, 1]
                auprc_score = average_precision_score(y_test, y_prob)
                # update label
                ax.get_lines()[-1].set_label(f"{base_label} (AUPRC = {auprc_score:.3f})")
            else:
                # keras model
                y_score = mod.predict(X_test)
                if y_score.shape[1] > 1:
                    y_score = y_score[:, 1]

                precision, recall, _ = precision_recall_curve(y_test, y_score)
                auprc_score = average_precision_score(y_test, y_score)
                ax.plot(recall, precision, color=current_color, label=f"{base_label} (AUC = {auprc_score:.3f})")

        if title is None: title = "Precision-Recall Curves"
        plt.grid(grid)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=fontsize)
        plt.ylabel("Precision", fontsize=fontsize)
        plt.title(title, fontsize=titlesize)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc=legend_loc)
        plt.tight_layout()
        if savepath: plt.savefig(savepath, dpi=300)
        plt.show()

    def prediction_confidence(self, model: Union[list, BaseEstimator, keras.Model] = None,
                              test_val: pd.DataFrame = None, test_val_labels: pd.DataFrame = None, title: str = None,
                              color: tuple = ("skyblue", "salmon"), labels: tuple[str] = None, kde: bool = False,
                              legend_loc: str = None, figsize: tuple[float, float] = (10, 5), savepath: str = None):
        global predict_inactive, predict_active
        if test_val is None:
            test_val = self.X_test
        if test_val_labels is None:
            label_val = self.y_test

        # prep dataset
        input_labels = label_val.squeeze()

        # Separate the features based on the labels
        feat_inactive = test_val[input_labels == 0]
        feat_active = test_val[input_labels == 1]

        if isinstance(model, BaseEstimator) or isinstance(model, keras.Model):
            # check model prediction type
            if hasattr(model, 'predict_proba'):
                # if model uses proba
                predict_inactive = model.predict_proba(feat_inactive)
                predict_active = model.predict_proba(feat_active)
            elif hasattr(model, 'predict'):
                predict_inactive = model.predict(feat_inactive)
                predict_active = model.predict(feat_active)
            else:
                # Raise an error if neither method is available
                raise ValueError("Cannot predict with model!")

        # plot results
        sns.histplot(predict_inactive, bins=50, kde=kde, color=color[0], label=labels[0], alpha=0.7)
        sns.histplot(predict_active, bins=50, kde=kde, color=color[1], label=labels[1], alpha=0.7)

        if title is None:
            title = "Prediction Confidence"

        # Add titles, labels, and legend
        plt.title(title)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.legend()  # todo fix figure legend
        plt.show()
        plt.tight_layout()
        figure = plt.gcf()  # get figure

        # save figure
        if savepath:
            plt.savefig(savepath, dpi=300)

        plt.close(figure)  # close figure
        return figure


def plot_enrichment(data: pd.DataFrame = None, rank_col: Union[str, list] = None, true_col: str = None,
                    title: str = None, label: Union[str, list] = None, color: Union[str, list] = None,
                    fill: bool = False, figsize: tuple[float, float] = (8, 8), savefig: str = None,
                    top_x=None, **kwargs):
    """
    Generate an enrichment curve plot. This can be used in conjunction with docking results or other methods that
    will rank molecules before final selection.
    :param data: pd.DataFrame
        A DataFrame containing molecules, their rank, and their actual label.
    :param rank_col: Union[str, list]
        Indicate column containing the molecule rank.
    :param true_col: str
        Indicate column containing the true label.
    :param title: str
        Title of the plot. If None is given, the default title will be used.
    :param label: Union[str, list]
        Label of the lines. If None is given, the default label will be used.
    :param color: Union[str, list]
        Set color scheme for the model and random line in plot. Can be a single string, which modifies the model
        color, or a list, which will modify both the model and the random line.
    :param fill: bool
        Determine whether to fill the area under the curve.
    :param figsize: tuple
        Set figure size.
    :param savefig: str
        Filepath to save figure.
    :param kwargs:
        A set of kwargs associated with plt.plot()
    :return:
    """
    # check color input
    global rank_list
    # convert rank_col input into a list if needed
    if isinstance(rank_col, str):
        rank_col = [rank_col]
        # print(rank_list) # check

    # set color palette
    if isinstance(color, str):
        line_color = [color]
        random_color = 'black'
    elif isinstance(color, list):
        line_color = color
        random_color = color[-1]
    else:
        line_color = ['cornflowerblue', 'lightcoral', 'seagreen']
        random_color = 'black'
    if len(rank_col) > len(line_color):
        raise ValueError("Input models are more than 3! Default color palette available for 3 models.")

    # check label input
    if isinstance(label, str):
        line_label = [label]
        random_label = 'Random'
    elif isinstance(label, list):
        line_label = label
        random_label = 'Random'
    else:
        line_label = rank_col
        random_label = 'Random'

    if title is None:
        title = "Enrichment Curves"

    """figure plot in a loop"""
    # plot curve
    plt.figure(figsize=figsize)

    for model, color, label in zip(rank_col, line_color, line_label):
        # sort table by rank in descending order
        data = data.sort_values(by=model, ascending=False)

        # calculate actives and total datasets
        total = np.cumsum(data[true_col])
        total_active = data[true_col].sum()
        total_dataset = np.arange(1, len(data) + 1)

        # convert values to percentage
        total_active_percentage = total / total_active * 100
        total_screened_percentage = total_dataset / len(data) * 100

        # plot curve
        plt.plot(total_screened_percentage, total_active_percentage, label=model, color=color, **kwargs)
        if fill:
            plt.fill_between(total_screened_percentage, total_active_percentage, color=color, alpha=0.1, **kwargs)

    # plot diagonal line
    # slope = 100 / data.shape[0] # to get slope if axes are not in 100%
    slope = 100 / 100
    plt.axline((0, 0), slope=slope, linestyle='--', label=random_label, color=random_color, alpha=0.2, **kwargs)

    # # if converted to whole numbers instead of percentage, ylimit will need to be dynmaically determined
    # ylimit = int(total_active)
    # xlimit = total_active
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    plt.xlabel('% of Molecules')
    plt.ylabel('% of Known Actives')
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.show()

    if savefig:
        plt.savefig(savefig, dpi=300)


def enrichment_factor(data: pd.DataFrame = None, rank_col: Union[str, list] = None, true_col: str = None,
                      top_percentage: float = 0.01, verbose: bool = False):
    """
    Generate an enrichment curve plot. This can be used in conjunction with docking results or other methods that
    will rank molecules before final selection.
    :param data: pd.DataFrame
        A DataFrame containing molecules, their rank, and their actual label.
    :param rank_col: Union[str, list]
        Indicate column containing the molecule rank.
    :param true_col: str
        Indicate column containing the true label.
    :param top_percentage: float
        Set the percentage to calculate enrichment factor.
    :param verbose: bool
        Set to give print statements from calculations.
    :return:):
    """
    # check color input
    global rank_list, dataset
    # convert rank_col input into a list if needed
    if isinstance(rank_col, str):
        rank_col = [rank_col]

    # constrain between 1 and 0
    if not (0 < top_percentage <= 1):
        raise ValueError("top_percentage must be between 0 and 1!")

    enrichment = []
    for model in rank_col:
        # sort table by descending order
        dataset = data.sort_values(by=model, ascending=False).reset_index(drop=True)

        # get total actives and compounds
        total_actives = len(dataset[dataset[true_col] == 1])
        total_compounds = len(dataset)

        # calculate number of compounds in top_percentage
        top_compounds = int(np.ceil(top_percentage * total_compounds))

        # calculate actives in the top_compounds
        top_actives = dataset[true_col].iloc[:top_compounds].sum()

        # calculate enrichment factor
        ef = (top_actives / total_actives) * (total_compounds / top_compounds)
        enrichment.append(ef)

    # add print statements
    if verbose:
        for model, ef in zip(rank_col, enrichment):
            print(f"Enrichment Factor for Top {top_percentage * 100}%:")
            print(f"{model}: {ef}")

    return dict(zip(rank_col, enrichment))


def ConfusionMatrix(y_true: np.ndarray = None, y_pred: np.ndarray = None, class_names: list = None,
                    cbar: bool = True, cmap: str = "Blues"):
    """Support function to plot matrix with number and percentages"""
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = ['Inactive', 'Active']
    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # plot matrix
    ax = sns.heatmap(cm, annot=labels, fmt="", cmap=cmap, cbar=cbar)

    # set tick alignment
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names, rotation=360)

    return ax


def ClassificationReport(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None, support: bool = None,
                         cbar: bool = True, cmap: str = "Blues"):
    """Support function to plot the classification report heatmap"""

    if class_names is None:
        class_names = ['Inactive', 'Active']

    # get report as dicta nd convert to df
    cr_dict = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
    cr_df = pd.DataFrame(cr_dict).transpose()
    # filter
    cr_df = cr_df.loc[class_names]
    if not support:
        cr_df = cr_df.drop(columns=['support'])

    # # for debugging
    # print(cr_df)

    # custom string annotation
    annot_labels = np.empty_like(cr_df.values, dtype=object)
    for i, row in enumerate(cr_df.index):
        for j, col in enumerate(cr_df.columns):
            val = cr_df.loc[row, col]
            # clean integer string
            if col == 'support':
                annot_labels[i, j] = f"{int(val)}"
            else:
                annot_labels[i, j] = f"{val:.3f}"

                # 2. Create a separate DataFrame for Seaborn to use for shading (The COLORS)
    # shade support column
    if support and 'support' in cr_df.columns:
        # scale column down to use lighter shades of cmap
        max_support = cr_df['support'].max()
        if max_support > 0:
            cr_df['support'] = (cr_df['support'] / max_support) * 0.3
        else:
            cr_df['support'] = 0.0

    # plot heatmap
    ax = sns.heatmap(cr_df, annot=annot_labels, fmt='', cmap=cmap, vmin=0, vmax=1.0, cbar=cbar, linewidths=2,
                     linecolor='white', cbar_kws={'label': 'Score'} if cbar else None)

    # set ticks
    ax.tick_params(axis='both', which='major', labelsize=10)

    return ax


if __name__ == "__main__":
    import doctest

    doctest.testmod()
