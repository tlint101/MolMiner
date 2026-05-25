"""
Scripts to import all sklearn models
"""

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
import warnings
import pickle
import ast
import os
from typing import Union
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

__all__ = ["Models", "Lazy"]

MODEL = namedtuple("MODEL", ["short_name", "long_name", "model_class"])
lr = MODEL("lr", "logistic regression", LogisticRegression)
rf = MODEL("rf", "random forest", RandomForestClassifier)
knn = MODEL("knn", "k-nearest neighbors", KNeighborsClassifier)
svc = MODEL("svc", "support vector", SVC)
xgb = MODEL("xgb", "xgboost", XGBClassifier)

# List of model aliases and corresponding model classes
MODEL_LIST = [lr, rf, knn, svc, xgb]


class Models:
    def __init__(
            self,
            features: pd.DataFrame = None,
            labels: pd.DataFrame = None,
            model: str = None,
    ):
        self.features = features
        self.labels = labels
        self.model = model

    def get_sk_model(
            self,
            model: Union[str, BaseEstimator] = None,
            hyperparams: Union[dict] = None,
            **kwargs
    ):
        """
        Function to obtain a given model from the namedtuple
        :param model: str
            Name of the model to get. Can be the long name or the short name.
        :param hyperparams: Union[str, dict]
            Set hyperparams for a given model. Must be a dictionary containing the param conditions or a string for a
            filepath to a .txt file containing hyperparam conditions. This can change depending on the model and should
            be checked on the scikit-learn site: https://scikit-learn.org/stable/
        :return:
        """

        # Get model from __init__ if specified
        if model is None:
            model = self.model

        # Obtain model from namedtuple
        model_lower = model.lower()

        # pull model from namedtuple list
        pulled_model = self._pull_from_namedtumple(model_lower)

        # if model is found in namedtuple_list
        if pulled_model is not None:
            # if hyperparams file path given
            if isinstance(hyperparams, str):
                # set model short name to model to obtain hyperparam dictionary and unpack
                model = pulled_model
                params = self._hyperparams(hyperparams)
                return pulled_model(**params)
            elif isinstance(hyperparams, dict):
                model = pulled_model
                return pulled_model(**hyperparams)
            else:
                # return default model
                return pulled_model()
        # if classifier is other than found in MODEL_LIST
        else:
            classifiers = {name: estimator for name, estimator in all_estimators(type_filter='classifier')}
            if model not in classifiers:
                raise ValueError(
                    f"Model '{model}' is not supported. Choose from: \n {list(classifiers.keys())}")

            return classifiers[model](**kwargs)

    def _hyperparams(self, filepath=None):
        """Support function for params for get_sk_model()"""
        # Read hyperparam file
        try:
            with open(filepath, "r") as file:
                dict = file.read().strip()
        except:
            raise ValueError(f"Current path, '{filepath}' not found! Give filepath param_path parameter!")

        # Convert string into dictionary
        params = ast.literal_eval(dict)

        return params

    def _pull_from_namedtumple(self, model_query):
        """support function to get model from the namedtuple"""
        # Loop through the list of classifiers and match with the model
        for classifier in MODEL_LIST:
            try:
                if model_query == classifier.short_name or model_query == classifier.long_name or model_query == classifier.model_class:
                    return classifier.model_class
            except:
                raise ValueError(f"Model '{model_query}' is not supported.")

    @staticmethod
    def predict(model: Union[BaseEstimator, XGBClassifier], query: pd.DataFrame = None,
                feat_col: Union[str, list] = 'fp'):
        """
        Function to make predictions for a given skmodel. Input should be a pd.DataFrame containing the indicated
        features for predictions.
        :param model: BaseEstimator
            The BaseEstimator from sklearn to be used for prediction.
        :param query: Pd.DataFrame
            The query molecules for prediction. Format should be as a pd.DataFrame. Given table should contain
        :param feat_col: Union[str, list]
            Indicate which columns are features. Can be a single column in a np.array format or a list of columns.
        """

        # if features is only a string because features or a fingerprint in an array
        global expanded_fp
        if type(feat_col) == str:
            # keep only fp column
            fp = query[[feat_col]]
            # expand fp array into individual columns
            expanded_fp = pd.DataFrame(fp[feat_col].tolist(), index=fp.index)
            # give column labels that matches with the column header
            expanded_fp.columns = [f'{feat_col}_{i + 0}' for i in range(expanded_fp.shape[1])]

        # if features contain additional stuff besides fingerprint
        elif type(feat_col) == list:
            # only keep feature columns
            expanded_fp = query[feat_col]

        # make predictions
        predictions = model.predict(expanded_fp)
        prediction_table = pd.DataFrame(predictions, columns=["predictions"])

        # map predictions table back to query table
        prediction_result = pd.concat([query, prediction_table], axis=1)

        return prediction_result

    # save or load models
    def save_model(self, model=None, savepath: str = None):
        """
        Save model as pickle file.
        :param model:
            Trained model to be saved.
        :param savepath:
            Filename for model to be saved.
        :return:
        """

        with open(savepath, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, filepath: str = None):
        """
        Read saved pickle file of model.
        :return:
        """
        _, ext = os.path.splitext(filepath)

        if ext in [".pkl", ".pickle"]:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
        elif ext in [".json", ".ubj", ".model"]:
            model = XGBClassifier()
            model.load_model(filepath)
        else:
            raise ValueError(f"Model '{filepath}' is not found!")
        return model


class Lazy:
    def __init__(self, features, labels, cv=10):
        self.features = features
        self.labels = labels
        self.cv = cv

    def get_lazy_scores(self, model_num=0, verbose=False):
        """
        Function to obtain lazy scores for all classification models in SKlearn. This will run through 32 different
        Classification models.
        :param model_num: str
            Set the number of models to test. Defaults to 0 which is all models.
        :param verbose: bool
            Add option for verbosity. This will tell which model is being looped
        """
        global name
        # Ignore convergence and runtime warnings
        warnings.filterwarnings("ignore")

        # # to print a list of all classification models from sklearn
        # estimators = all_estimators(type_filter='classifier')
        # i = 0
        # for name, class_ in estimators:
        #     print(f'{i}. {class_.__name__}')
        #     i += 1

        # Place all classifiers into a dictionary and model names in list
        estimators = all_estimators(type_filter="classifier")
        model_dict = {name: class_ for name, class_ in estimators}

        # if want to test a smaller set, input model_num
        if model_num > 0:
            num_of_models = min(model_num, len(model_dict))
            # convert to list of keys to sample safely
            keys = list(model_dict.keys())
            selected_keys = random.sample(keys, num_of_models)
            model_dict = {k: model_dict[k] for k in selected_keys}

        # # For troubleshooting, print names of all model class
        # print("this is model_name\n", model_name)
        # print("this is the dictionary\n",model_dict)

        # These require multilabels or other junk that may have issues with our data.
        models_to_remove = [
            "CategoricalNB",
            "ClassifierChain",
            "FixedThresholdClassifier",
            "MultiOutputClassifier",
            "OneVsOneClassifier",
            "OneVsRestClassifier",
            "OutputCodeClassifier",
            "RadiusNeighborsClassifier",
            "StackingClassifier",
            "TunedThresholdClassifierCV",
            "VotingClassifier",
            "SelfTrainingClassifier",
            "QuadraticDiscriminantAnalysis"
        ]

        # # for troubleshooting. Only lazily train a few models for quicker testing.
        # from itertools import islice
        # model_dict = dict(islice(model_dict.items(), 2))

        # Remove items from the dictionary based on the keys
        for key in models_to_remove:
            if key in model_dict:
                del model_dict[key]

        # add xgboost model
        if "XGBClassfier" not in model_dict:
            model_dict['XGBClassifier'] = XGBClassifier

        # Loop through models
        print(f"{len(model_dict)} Models to loop through I guess...")
        scoring_metrics = []

        for name, model_class in tqdm(model_dict.items(), desc=f"Lazily going through models..."):
            try:
                if verbose is True:
                    print(f"Looping through {name}")

                # instantiate the model
                if "XGBClassifier" in name:
                    model_instance = model_class(n_jobs=1, tree_method='exact')
                else:
                    model_instance = model_class()

                scores = self._get_cv_scores(X_features=self.features, y_labels=self.labels, model=model_instance,
                                             cv=self.cv)
                scoring_metrics.append(scores)
            except Exception as e:
                print(f"Error while processing: {e}")
                continue

        # Generate output as DataFrame
        output_df = pd.DataFrame(scoring_metrics)

        return output_df

    def _get_cv_scores(self, X_features, y_labels, model, cv=5):
        """
        Internal function for lazy classification. Obtain Cross Validation Scores for a given model. This is a custom
        script created to fit the .csv due to my molecular splits. Models will output the model, an array of CV scores and the CV mean.

        :param X_features:
        :param y_labels:
        :param model:
        :param cv:
        :return:
        """

        global feat_test, label_predict, label_test
        # skmodel = Models()
        # model = skmodel.get_sk_model(model)
        # model_name = model.__class__.__name__

        # get CV
        cv_index = self._get_cv_split_skmodels(y_label=y_labels, cv=cv)

        # To get columns containing all CV* header
        drop_cols = y_labels.filter(regex="CV*").columns
        y_label = y_labels.drop(columns=drop_cols)
        X_features = X_features.drop(columns=drop_cols)

        # convert y_label to numpy array for ML model
        y_labels = np.ravel(y_label).astype(int)
        X_features = X_features.values.astype(np.float32)

        # dict to hold results
        results_dict = {"Model": model.__class__.__name__}

        metrics = {
            "Accuracy": "accuracy",
            "F1 Score": "f1",
            "Recall": "recall",
            "Precision": "precision",
            "MCC": "matthews_corrcoef",
            "ROC AUC": "roc_auc"
        }

        for metric_name, scoring_param in metrics.items():
            try:
                scores = cross_val_score(model, X_features, y_labels, cv=cv_index, scoring=scoring_param)
                results_dict[metric_name] = scores
                results_dict[f"{metric_name}_avg"] = np.mean(scores)
            except Exception as e:
                results_dict[metric_name] = np.nan
                results_dict[f"{metric_name}_avg"] = np.nan

        # reorder cols with avg cols at the end
        for metric_name in metrics.keys():
            if isinstance(results_dict[metric_name], np.ndarray):
                results_dict[f"{metric_name}_avg"] = np.mean(results_dict[metric_name])
            else:
                results_dict[f"{metric_name}_avg"] = np.nan

        # reorder cols
        column_template = ["Model", "Accuracy", "F1 Score", "Recall", "Precision", "MCC", "ROC AUC", "Accuracy_avg",
                           "F1 Score_avg", "Recall_avg", "Precision_avg", "MCC_avg", "ROC AUC_avg"]
        results_dict = {key: results_dict[key] for key in column_template}

        return results_dict

    def _get_cv_split_skmodels(self, y_label, cv=5):
        """
        Support function for get_cv_score. Obtain index for train/test split. Index is based on input CV label headers
        from DataFrame. This will output a tuple with the indices of the train and test split based on CV label headers.

        :param y_label:
        :param cv:
        :return:
        """

        splits = []

        for i in range(cv):
            header = f"CV{i}"

            train_indexes = y_label[y_label[header] == "train"].index.values.astype(int)
            test_indexes = y_label[y_label[header] == "test"].index.values.astype(int)
            splits.append((train_indexes, test_indexes))

        return splits


if __name__ == "__main__":
    import doctest

    doctest.testmod()
