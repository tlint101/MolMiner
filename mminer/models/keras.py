"""
Scripts for Keras model wrapper
"""
# Set keras backed dependencies and MacOS MPS Fallback to CPU if needed
import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras.src.regularizers import L2
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Input, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from typing import Union, Optional
import numpy as np
import random

__all__ = ["Keras"]


# set default device
def _set_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # set MPS fallback
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    else:
        return "cpu"


DEFAULT_DEVICE = _set_default_device()


def set_device(device: str = None):
    """
    Manually overide default device
    :param device: str
        Device can be set to 'cuda', 'cpu', or 'mps'. If None, returns the current device set by defualt.
    :return:
    """
    global DEFAULT_DEVICE
    if device is not None:
        DEFAULT_DEVICE = device
        # Set MPS fallback if needed
        if device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return DEFAULT_DEVICE


def get_device():
    """
    Get teh current default device
    :return:
    """
    return DEFAULT_DEVICE


def set_backend(backend: str = 'torch'):
    """
    Set keras 3.0 backend. Only accepts 'torch',
    :param backend:
    :return:
    """
    if backend == 'torch':
        os.environ["KERAS_BACKEND"] = "torch"
    elif backend == 'jax':
        os.environ["KERAS_BACKEND"] = "jax"
    elif backend == 'tf':
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
    else:
        raise ValueError("Only support 'torch', 'jax', 'tf'!")


# set seed
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if cuda is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


class Keras:
    def __init__(self, model: Optional = None, seed: Optional[int] = None):
        if model is None:
            self.model = Sequential()
        else:
            self.model = model
        self.model_history = None
        self.is_compiled = False
        # set seed
        if seed is None:
            self.seed = 42
        else:
            self.seed = seed
        # set seed globally
        set_seed(self.seed)

    def add_layer(self, layer_type, *args, **kwargs):
        """
        Function to add a keras layers.

        :param layer_type:
        :param args:
        :param kwargs:
        :return:
        """
        if layer_type == "Input":
            self.model.add(Input(*args, **kwargs))
        elif layer_type == "Dense":
            self.model.add(Dense(*args, **kwargs))
        elif layer_type == "Dropout":
            self.model.add(Dropout(*args, **kwargs))
        else:
            raise ValueError(
                f"Only 'Input', 'Dense', and 'Dropout' layers are supported!"
            )

    def summary(self, save_path: str = None):
        """
        Print model summary.
        :param save_path: str
            File path to save the figure. If None, only the summary will be printed.
        :return:
        """

        # custom function to save model.summary as separate txt file
        def _save_summary(s, save_path=save_path):
            with open(save_path, "w") as f:
                print(s, file=f)

        # output model summary
        if save_path is None:
            return self.model.summary()
        else:
            return self.model.summary(print_fn=_save_summary)

    def evaluate(
            self,
            x=None,
            y=None,
            batch_size=None,
            verbose="auto",
            sample_weight=None,
            steps=None,
            callbacks=None,
            return_dict=False,
            **kwargs,
    ):
        """
        Wrapper function to return the loss value & metrics values for the model in test mode. Additional info can be
        found here: https://keras.io/api/models/model_training_apis/#evaluate-method

        :param x: Input data.
        :param y: Target data.
        :param batch_size: Number of samples per batch of computation. If unspecified, will default to 32.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. "auto" becomes 1 for most cases.
        :param sample_weight: Optional NumPy array of weights for the test samples, used for weighting the loss
        function.
        :param steps: Total number of steps (batches of samples) before declaring the evaluation round finished.
        :param callbacks: List of callbacks to apply during evaluation.
        :param return_dict: If True, loss and metric results are returned as a dict, with each key being the name of the
        metric. If False, they are returned as a list.
        :param kwargs:
        :return:
        """

        evaluate = self.model.evaluate(
            x=x,
            y=y,
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
            return_dict=return_dict,
            **kwargs,
        )

        return evaluate

    def compile(self, **kwargs):
        """
        Compile Keras model.
        :param kwargs: https://keras.io/api/models/model_training_apis/
        :return:
        """
        # set fallback
        self.model.to(DEFAULT_DEVICE)

        # compile
        self.model.compile(**kwargs)
        self.is_compiled = True

    def fit(self, X, y, **kwargs):
        """
        Fit data to Keras model.

        :param X: Feature as a numpy array or tensor.
        :param y: Labels as a numpy array or tensor.
        :param kwargs: https://keras.io/api/models/model_training_apis/#fit-method
        :return:
        """
        if not self.is_compiled:
            raise RuntimeError("Model needs to be compiled first!")
        history = self.model.fit(X, y, **kwargs)

        return history

    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None):
        """
        Wrapper function to predict from the Keras model.
        """

        predictions = self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks)
        return predictions

    def save(self, save_path: str = None):
        """
        Wrapper function to save the Keras model.
        """
        print(self.model)
        self.model.save(save_path)

    def basic_mlp(
            self,
            features=None,
            labels=None,
            batch_size: int = 128,
            epochs: int = 20,
            validation_split: float = 0.2,
            validation_data=None,
            summary: bool = False,
    ):
        """
        This function is for basic MLP classification model for small-molecules. Further reading on determining the
        number of layers to use: https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3. Information
        on the keras.fit() method can be found here: https://keras.io/api/models/model_training_apis/

        :param features: np.array or pd.DataFrame
            Input features. Can be a Datatable or a numpy array.
        :param labels: np.array or pd.DataFrame
            Input labels. Can be a Datatable or a numpy array.
        :param batch_size: int
            The number of samples per gradient update. If unspecified, batch_size will default to 128
        :param epochs: int
            The number of epochs to train the model.
        :param validation_split: float
            Fraction of the training data to be used as validation data.
        :param validation_data: tuple
            Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be
            trained on this data. validation_data will override validation_split. When used, must be a tuple like ->
            (x_test, y_test).
        :param summary: bool
            Print out a string summary of the network. If set to True, only the summary will be output and the model
            will not be fit.
        :return:
        """
        # Get shape from features
        shape = features.shape[1]

        # Build model
        self.model = keras.Sequential(
            [
                Input(shape=(shape,)),
                Dense(units=64, activation="relu", kernel_regularizer=L2(0.01)),
                Dropout(rate=0.5),
                Dense(units=32, activation="relu", kernel_regularizer=L2(0.01)),
                Dropout(rate=0.5),
                Dense(units=1, activation="sigmoid"),
            ]
        )

        if summary is True:
            return self.model.summary()

        basic_model = self.model

        basic_model.to(DEFAULT_DEVICE)

        # Compile model
        basic_model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        self.is_compiled = True

        callback = EarlyStopping(monitor="val_loss", patience=15, verbose=1)

        # fit model
        if validation_data:
            model_history = basic_model.fit(
                features,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=[callback],
            )
        else:
            model_history = basic_model.fit(
                features,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=[callback],
            )

        self.model_history = model_history

        return model_history

    def plot_model(
            self,
            model: Optional[keras.models.Model] = None,
            save_path: str = None,
            show_shapes: bool = True,
            show_dtype: bool = True,
            show_layer_names: bool = True,
            rankdir: str = "TB",
            expand_nested: bool = False,
            dpi=300,
            show_layer_activations=False,
            show_trainable=False,
    ):
        """
        Convert Keras model into a dot format to visualize the model layers.
        Note: The figure will, by default, be generated and saved in the same default folder as the file. The issue is
        due to Keras backend. The figure will have to be deleted manually if not needed.

        :param save_path: str
            File path to save the figure. If None, figure will only be drawn.
        :param show_shapes: bool
            To display the layer shape.
        :param show_dtype: bool
            To display the layer dtype.
        :param show_layer_names: bool
            To display the layer names.
        :param rankdir: str
            Specify the format of the plot. "TB" creates a vertical plot. "LR" creates a horizontal plot.
        :param expand_nested: bool
            To expand nested layers.
        :param dpi: int
            Set image resolution.
        :param show_layer_activations: bool
            Display layer activation. Only use for layers with 'activation' property.
        :param show_trainable: bool
            Display if a layer is trainable.
        :return:
        """
        if model is None:
            model = self.model

        # Define common parameters
        plot_params = {
            "show_shapes": show_shapes,
            "show_dtype": show_dtype,
            "show_layer_names": show_layer_names,
            "rankdir": rankdir,
            "expand_nested": expand_nested,
            "dpi": dpi,
            "show_layer_activations": show_layer_activations,
            "show_trainable": show_trainable,
        }

        # Conditional for saving figure
        if save_path is not None:
            plot_params["to_file"] = save_path

        # plot model layers
        fig = keras.utils.plot_model(model=model, **plot_params)

        return fig

    def plot_histroy(
            self,
            model_history,
            metrics: Union[str, list] = ["accuracy", "loss"],
            plot: int = 2,
            legend_loc: str = "best",
            save_path: str = None,
            line_color=("#1f77b4", "#ff7f0e"),
            title: str = "Keras Training History",
    ):
        """
        Function to plot the history of the keras model. This will generate a single plot containing 2 subplots.

        :param model_history: keras model that has been fitted.
        :param metrics: list
            Give a list or string of metrics to plot.
        :param plot: int
            Set the number of plots to generate.
        :param save_path: file path to save the plot.
        :param line_color: A tuple containing the color of the lines.
        :param title: A string for the suptitle.
        :return:
        """

        # obtain metrics in a list
        global fig
        import seaborn as sns
        sns.set_style("white")

        metric_list = []
        for key in model_history.history.keys():
            metric_list.append(key)

        # convert metrics from str to list
        if isinstance(metrics, str):
            input_metrics = [metrics]
        else:
            input_metrics = metrics

        metric_input_match = [metric for metric in input_metrics if metric in metric_list]

        metrics_len = len(metric_input_match)
        if metrics_len == 0:
            raise ValueError("Input metrics not found!")

        # get plot number
        if plot is None:
            plot = 2
        elif metrics_len == 2:
            plot = 2
        elif metrics_len % 2 == 0 and metrics_len == 4:
            plot = 4
        elif metrics_len > 4:
            raise ValueError("Can only create 4 plots in a grid!")
        elif metrics_len == 1:
            pass
        elif metrics_len % 2 != 0:
            raise ValueError("Can only plot even number of metrics! Try plotting them individually!")
        else:
            raise ValueError("Issue with the plot number!")

        # plot figure
        if len(metric_input_match) == 2 and plot == 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(title)
            axes[0].plot(model_history.history[metric_input_match[0]], color=line_color[0])
            axes[0].plot(model_history.history['val_' + metric_input_match[0]], color=line_color[1])
            axes[0].legend(["Train", "Test"], loc=legend_loc)
            axes[0].set_title(f"Model {metric_input_match[0].capitalize()}")
            axes[0].set_ylabel(metric_input_match[0].capitalize())
            axes[0].set_xlabel("Epoch")
            axes[0].grid(False)

            axes[1].plot(model_history.history[metric_input_match[1]], color=line_color[0])
            axes[1].plot(model_history.history['val_' + metric_input_match[1]], color=line_color[1])
            axes[1].legend(["Train", "Test"], loc=legend_loc)
            axes[1].set_title(f"Model {metric_input_match[1].capitalize()}")
            axes[1].set_ylabel(metric_input_match[1].capitalize())
            axes[1].set_xlabel("Epoch")
            axes[1].grid(False)

        elif plot % 2 == 0 and plot > 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            fig.suptitle(title)
            axes[0, 0].plot(model_history.history[metric_input_match[0]], color=line_color[0])
            axes[0, 0].plot(model_history.history['val_' + metric_input_match[0]], color=line_color[1])
            axes[0, 0].legend(["Train", "Test"], loc=legend_loc)
            axes[0, 0].set_title(f"Model {metric_input_match[0].capitalize()}")
            axes[0, 0].set_ylabel(metric_input_match[0].capitalize())
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].grid(False)

            axes[0, 1].plot(model_history.history[metric_input_match[1]], color=line_color[0])
            axes[0, 1].plot(model_history.history['val_' + metric_input_match[1]], color=line_color[1])
            axes[0, 1].legend(["Train", "Test"], loc=legend_loc)
            axes[0, 1].set_title(f"Model {metric_input_match[1].capitalize()}")
            axes[0, 1].set_ylabel(metric_input_match[1].capitalize())
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].grid(False)

            axes[1, 0].plot(model_history.history[metric_input_match[2]], color=line_color[0])
            axes[1, 0].plot(model_history.history['val_' + metric_input_match[2]], color=line_color[1])
            axes[1, 0].legend(["Train", "Test"], loc=legend_loc)
            axes[1, 0].set_title(f"Model {metric_input_match[2].capitalize()}")
            axes[1, 0].set_ylabel(metric_input_match[2].capitalize())
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].grid(False)

            axes[1, 1].plot(model_history.history[metric_input_match[3]], color=line_color[0])
            axes[1, 1].plot(model_history.history['val_' + metric_input_match[3]], color=line_color[1])
            axes[1, 1].legend(["Train", "Test"], loc=legend_loc)
            axes[1, 1].set_title(f"Model {metric_input_match[3].capitalize()}")
            axes[1, 1].set_ylabel(metric_input_match[3].capitalize())
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].grid(False)

            fig.subplots_adjust(hspace=0.3)

        elif isinstance(metrics, str):  # plot single figure
            val_metric = "val_" + metrics
            plt.plot(model_history.history[metrics])
            plt.plot(model_history.history[val_metric])
            plt.title(f"Model {metrics.capitalize()}")
            plt.ylabel(metrics.capitalize())
            plt.xlabel("Epoch")
            plt.grid(False)
            if metrics == "accuracy":
                plt.legend(["train", "test"], loc=legend_loc)
            else:
                plt.legend(["train", "test"], loc=legend_loc)

        if save_path is not None:
            fig.savefig(save_path, dpi=300)

        plt.show()

    def load_model(self, path: str = None):
        """
        Read saved keras model from a keras file.
        :return:
        """
        model = keras.saving.load_model(filepath=path)
        return model


if __name__ == "__main__":
    import doctest

    doctest.testmod()
