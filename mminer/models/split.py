"""
Scripts to extract information from a pd.DataFrame. Data should be split previously using the utils.split() module.
Following scripts will present CV splitting and CV scoring based on the pd.DataFrame input
"""
import keras
import re
import pandas as pd
import numpy as np
import splito
from typing import Optional, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import auc, precision_recall_curve, f1_score, recall_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score
from mminer.models.skmodels import Models
from mminer.models.utils import Splitting_utils

__all__ = ["Splitter", "CrossVal"]

"""
Split Dataset using different splitting criteria for CV scoring.
"""


class Splitter:
    def __init__(self, data: pd.DataFrame = None):
        self.data: pd.DataFrame = data
        self._model = None

    @staticmethod
    def MOODSplitter_list():
        print(
            "Splitters Tested: Random, Scaffold, Perimeter, MaxDissimilarity, MolecularMinxMaxSplit, MolecularWeight, and KMeansSplit."
        )

    def MOODSplitter(
            self,
            splitters: Optional[dict] = None,
            smiles: np.ndarray = None,
            features: np.ndarray = None,
            label: np.ndarray = None,
            library_feat: np.ndarray = None,
            test_size: float = 0.2,
            n_splits: int = 10,
            n_clusters: int = 10,
            cutoff: float = 0.5,
            seed: int = 42,
            verbose: bool = False,
    ):
        """
        Split the dataset according to MOODProtocol. This will test different type of splitters.
        :param splitters: dict
            Give option for specific of splitters. If none is given, all spliters will be used.
        :param smiles: np.ndarray
            An array of smiles from the training/test set.
        :param features: np.ndarray
            An array of fingerprints from the training/test set.
        :param label: np.ndarray
            An array of activity labels from the training set
        :param library_feat: np.ndarray
            An array of fingerprints from the library to be screened.
        :param test_size: float
            The size of the test set.
        :param n_splits: int
            The number of splits to be used.
        :param n_clusters: int
            The number of clusters to be generated. Used for Umap splits.
        :param cutoff: float
            Similarity cutoff for each Butina split.
        :param seed: int
            The random state to be used.
        :param verbose: bool
            Set progress bar for MOODSplitter
        :return:
        """
        # Define the candidate splitters
        # Since we use the scikit-learn interface, this can also be sklearn Splitters
        if splitters is None:
            mapping = Splitting_utils.get_split_mapping(smi_col=smiles, test_size=test_size, n_splits=n_splits,
                                                        cutoff=cutoff, n_clusters=n_clusters, seed=seed)
            # extract function objects
            splitters = {key: value() for key, value in mapping.items()}

        # prime MOOD protocol
        splitters = splito.MOODSplitter(splitters, n_jobs=-1)

        # fit data to the MOOD protocol and save to instance variable
        splitters.fit(
            X=np.stack(features),
            X_deployment=np.stack(library_feat),
            y=label,
            progress=verbose,
        )

        # output best result as a pd.DataFrame
        best_result = splitters.get_protocol_results()

        return best_result

    @staticmethod
    def select_best_split(split_result):
        """
        Get the best split from the MOODSplitter() function as a variable. This will be used as the method in
         train_test(). This will output string, which will be used for the train_test_split() below.
        :param split_result: pd.DataFrame
            DataFrame result from MOODSplitter()
        :return: String of best split method.
        """
        # Optional: can replace with splitter.get_prescribed_splitter() for repeatability?
        best = max(split_result["representativeness"])
        best_split_row = split_result[split_result["representativeness"] == best]
        best_split = best_split_row.iloc[0]["split"]

        print("The best split method is:", best_split)
        return best_split

    def slice_data(self, data: pd.DataFrame = None, cols: Union[str, list] = None):
        """
        Slice input pd.DataFrame. The splitting for the columns will use the python "startswith()" format to extract
        columns. Can be used if user needs to include additional features besides molecular fingerprints
        :param data: pd.DataFrame = None
            Input pd.DataFrame to be sliced.
        :param cols: Union[str, list] = None
            Columns to be extracted. If a string is given, only columns matching with the string will be retained. Else,
            if a list is given, all columns matching list will be retained. Wildcard (*) at the end of the string is
            acceptable in string or list inptus.
        :return:
        """
        global cols_df, split_label_df
        if data is None:
            data = self.data

        # Get features. input is a str, convert to a list
        if isinstance(cols, str):
            cols_df = data.loc[:, data.columns.str.startswith(cols)]
            if cols_df.empty:
                raise ValueError('No matches to cols input!')

        elif isinstance(cols, list):
            # extract matching cols
            matched_cols = []
            for col_str in cols:
                # set potential wildcard matching
                pattern = col_str.replace('*', '.*')
                # match columns that fit pattern
                matched_cols.extend([col for col in data.columns if re.search(pattern, col, re.IGNORECASE)])

            # Remove duplicates and keep original order
            matched_cols = list(dict.fromkeys(matched_cols))

            if matched_cols:
                cols_df = data[matched_cols]
            else:
                raise ValueError("No matches to cols input!")

        return cols_df

    def extract_split(self, data: pd.DataFrame, feat_col: Optional[Union[str, list]] = None, label_col: str = None,
                      split_col: str = None):
        """
        For use in optimizing a model. Extract the splits from a dataframe labeled for train/test. Extract features and
        labels from the DataFrame and place into separate DataFrames.
        :param data: pd.DataFrame
            Input pd.DataFrame for splitting into train/test sets.
        :param feat_col: Optional[Union[str, list]]
            Columns headers containing features. Users can a list of column headers for the features. As the strings are
            filterd using "startswith()", a common string should suffice. Or for a specific feature column can be given
            as a single column header string. For example, 'fp_' to obtain feature columns labeled as 'fp_0', 'fp_1', etc.
        :param label_col: str
            Column containing the activity labels.
        :param split_col: str
            Column containing the train/test split labels.
        :return X_train, X_test, y_train, y_test
        """
        global feat_cols, X_train, X_test
        if data is None:
            data = self.data

        # Split data into train/test sets
        train = data[data[split_col] == "train"]
        test = data[data[split_col] == "test"]

        # Get features. input is a str, convert to a list
        if isinstance(feat_col, str):
            X_train = train.loc[:, data.columns.str.startswith(feat_col)]
            X_test = test.loc[:, data.columns.str.startswith(feat_col)]
        elif isinstance(feat_col, list):
            # extract matching cols
            matched_cols = []
            for col_str in feat_col:
                # set potential wildcard matching
                pattern = col_str.replace('*', '.*')
                # match columns that fit pattern
                matched_cols.extend([col for col in data.columns if re.search(pattern, col, re.IGNORECASE)])

            # Remove duplicates and keep original order
            matched_cols = list(dict.fromkeys(matched_cols))

            if matched_cols:
                X_train = train[matched_cols]
                X_test = test[matched_cols]
            else:
                raise ValueError("No matches to cols input!")

        # Get activity labels and flatten label array
        y_train = train.loc[:, train.columns.str.startswith(label_col)]
        # y_train = np.ravel(y_train) # previous method loses index
        y_train = y_train.squeeze()
        y_test = test.loc[:, test.columns.str.startswith(label_col)]
        # y_test = np.ravel(y_test) # previous method loses index
        y_test = y_test.squeeze()

        return X_train, X_test, y_train, y_test

    def train_test_split(self,
                         data: pd.DataFrame = None,
                         smi_col: str = None,
                         split: str = 'Random',
                         split_col: str = None,
                         test_size: float = 0.2,
                         cutoff: float = 0.65,
                         n_clusters: int = 10,
                         seed: int = 42):
        # todo add option for a val section
        # todo rename to just "split"?
        """
        Split the dataset into train/test labels. Many splitting options are available. By default, Random will be used.
        :param data: pd.DataFrame = None
            Unprocessed DataFrame containing data for model building.
        :param smi_col: str = None
            Name of SMILES column.
        :param split: str = None
            Name of the split method to use. The best split above can be used. Default set to 'Random'.
        :param split_col: str = None
            Name of column containing the train/test split labels. If None, will default to string in split param.
        :param test_size: float = 0.2
            Size of the test split.
        :param cutoff: float
            Similarity cutoff for each Butina split.
        :param n_clusters: int
            The number of clusters to be generated. Used for Umap splits.
        :param seed: int = 42
            Random seed to be used.
        :return: pd.DataFrame
        """
        global splits, split_col_name
        if data is None:
            data = self.data

        if smi_col is None:
            smi_col = 'smiles'

        # pull split
        mapping = Splitting_utils.get_split_mapping(smi_col=data[smi_col].values, test_size=test_size,
                                                    n_splits=1, cutoff=cutoff, n_clusters=n_clusters, seed=seed)
        if split not in mapping:
            raise ValueError(f"'{split}' is invalid! Available methods: {list(mapping.keys())}")

        # extract split
        splits = mapping[split]()

        train_idx, test_idx = next(splits.split(data[smi_col].values))
        assert train_idx.shape[0] > test_idx.shape[0]

        # Append standard numbering to CV labels
        if split_col is None:
            split_col_name = split.lower()
        data.loc[train_idx, f"{split_col_name}_split"] = "train"
        data.loc[test_idx, f"{split_col_name}_split"] = "test"

        self.data = data

        return data

    def repeated_kfold(self,
                       data: pd.DataFrame = None,
                       split_size: int = 5,
                       repeat: int = 5,
                       seed: int = 42):
        """
        Split the dataset using RepeatedKFold. The script is used to more easily generate 5X5 repeated CV as suggested by
        Ash et al. DOI: 10.1021/acs.jcim.5c01609
        Split the dataset into train/test labels. Many splitting options are available. By default, Random will be used.
        :param data: pd.DataFrame = None
            Unprocessed DataFrame containing data for model building.
        :param split_size: int = 5
            Set the size of the split for each seed.
        :param repeat: int = 5
            Number of times to repeat the split.
        :param seed: list = None
            Random seed to be used for each split. If set to None, will default to [42, 1701, 451, 88, 421].
        :return: pd.DataFrame
        """
        global splits, split_col_name
        if data is None:
            data = self.data

        # repeated kfold split
        kf = RepeatedKFold(n_splits=split_size, n_repeats=repeat, random_state=seed)

        # add col with train/test labels for each fold
        for i, (train_idx, test_idx) in enumerate(kf.split(data)):
            colname = f"CV{i}"
            data[colname] = "unused"  # init
            data.loc[train_idx, colname] = "train"
            data.loc[test_idx, colname] = "test"

        self.data = data

        return data

    # todo add script to split library for lead optimization
    # todo write a demo script for this, a model for lead optimization.
    def lo_splitter(self):
        pass

    def validation_set(self, data: Optional[pd.DataFrame] = None, activity_col: str = None,
                       active_percentage: float = 0.01, inactive_percentage: float = 0.1,
                       cutoff: Optional[float] = None, seed: Optional[int] = 42):
        """
        Extract a validation set. Data input must be in DataFrame format. Will default to binary classificaiton. If
        continuous data given, the cutoff param must be given as a float and indicate anything greater than or equal to.
        Table will be shuffled and a percentage of the dataset from both will be extracted based on the percentage
        param.  Finally, both tables will be combined and given as a final output.
        :param data: Optional[pd.DataFrame]
            Input data table.
        :param activity_col: str
            Indicate column containing activity label. Can be binary classification or continuous.
        :param active_percentage: float
            Set the percentage of the active dataset to be used for validation.
        :param inactive_percentage: float
            Set the percentage of the inactive dataset to be used for validation.
        :param cutoff: Optional[float]
            Set the cutoff based on activity labels. Only needed if the data is continuous.
        :param seed: Optional[int]
            Set the seed for shuffling the data.
        :return: pd.DataFrame
        """

        # check instance variable
        if data is None:
            data = self.data
        if seed is None:
            seed = None

        # if cutoff is continuous
        if cutoff:
            active = data[data[activity_col] >= cutoff]
            inactive = data[data[activity_col] < cutoff]
        # if cutoff is None, assume binary classification
        else:
            active = data[data[activity_col] == 1]
            inactive = data[data[activity_col] == 0]

        # shuffle table
        active = active.sample(frac=1, random_state=seed).reset_index(drop=True)
        inactive = inactive.sample(frac=1, random_state=seed).reset_index(drop=True)

        # get number of validation data
        active_num = len(active)
        inactive_num = len(inactive)
        active_val_num = np.ceil(active_num * active_percentage)
        inactive_val_num = np.ceil(inactive_num * inactive_percentage)

        # extract validation data
        active = active.head(int(active_val_num))
        inactive = inactive.head(int(inactive_val_num))

        validation_set = pd.concat([active, inactive], ignore_index=True)

        # remove the validation_set from the larger dataset
        validation_list = validation_set.iloc[:, 0].tolist()

        data = data[~data.iloc[:, 0].isin(validation_list)]
        data = data.reset_index(drop=True)

        # set data back to attribute
        self.data = data

        return validation_set


"""
Obtain splits for Cross-Validation.
"""


class CrossVal(Splitter):
    def __init__(self, data=None, model: str = None):
        super().__init__(data)
        self._X_features = None
        self._y_labels = None
        self._y_predict = None
        models = Models()
        if model:
            self._model = models.get_sk_model(model)
        else:
            self._model = None

    def show(self, rows: int = 5):
        """
        Show DataFrame of table for processing
        :param rows: int
            The number of rows to show.
        :return:
        """
        if rows:
            return self.data.head(rows)
        else:
            return self.data

    def cv_split(self, data: pd.DataFrame = None, smi_col: str = None, activity_col: str = None, split: str = 'Random',
                 test_size: float = 0.2, n_splits: int = 10, n_clusters: int = 10, cutoff: float = 0.65,
                 seed: Union[int, list] = 42):
        """
        Split dataset using Splito for use as CV validation.
        Split the dataset for training/testing. Method should be used after performing the MOODProtocol.
        :param data: pd.DataFrame
            Unprocessed DataFrame of training library.
        :param smi_col: str
            Name of SMILES column.
        :param activity_col: str
            Name of activity column. Needed for StratifiedShuffleSplit.
        :param split: str
            Name of the split method to use. The best split above can be used. Default set to 'Random'.
        :param test_size: int
            Size of test split.
        :param n_splits: int
            The number of splits to be used
        :param n_clusters: int
            The number of clusters to be generated. Used for Umap splits.
        :param cutoff: float
            The clustering cutoff for butina.
        :param seed: int or list
            For split reproducibility. Input can be a single int or a list of ints.
        :return: pd.DataFrame
        """

        if data is None:
            data = self.data
        if activity_col is None:
            activity_col = 'activity'

        # pull split
        mapping = Splitting_utils.get_split_mapping(smi_col=data[smi_col].values, test_size=test_size,
                                                    n_splits=n_splits, cutoff=cutoff, n_clusters=n_clusters, seed=seed)
        best_split = split

        if best_split not in mapping:
            raise ValueError(f"'{best_split}' is invalid! Available methods: {list(mapping.keys())}")

        if best_split == "MolecularWeight":
            print("No difference in CV splits with MolecularWeight!")

        splits = mapping[best_split]()

        if smi_col is None:
            smi_col = 'smiles'

        # extract train/test label and append to data
        for x, (train_idx, test_idx) in enumerate(splits.split(data[smi_col].values, data[activity_col].values)):
            # Append standard numbering to CV labels
            data.loc[train_idx, f'CV{x}'] = "train"
            data.loc[test_idx, f'CV{x}'] = "test"

        # set data with split info
        self.data = data

        return data

    def extract_cv(
            self,
            feat_col: Union[list, str] = None,
            label_col: str = None,
            cv: int = 5,
    ):
        """
        Extract features and labels from the DataFrame and place into separate DataFrame tables.
        :param feat_col: Union[list, str]
            Columns headers containing features. This will filter the input DataFrame using "startswith()". Column
            headers that match the string will be filtered. Use general string names. For example, 'fp_' to obtain
            feature columns labeled as 'fp_0', 'fp_1', etc.
        :param label_col: str
            Column containing the activity labels.
        :param cv: int
            The number of CV groups in table.
        """
        data = self.data.copy()

        # Include CV columns
        cv_num = []
        for i in range(cv):
            header = f"CV{i}"
            cv_num.append(header)

        cv_cols = data.filter(cv_num)

        # Get features
        if isinstance(feat_col, str):
            features = data.loc[:, data.columns.str.startswith(feat_col)]
            self._X_features = pd.concat([features, cv_cols], axis=1)
        elif isinstance(feat_col, list):
            # match columns with inputs from feat_col list
            matching_cols = [col for col in data.columns if any(col.startswith(feat) for feat in feat_col)]
            features = data[matching_cols]
            self._X_features = pd.concat([features, cv_cols], axis=1)

        # Get labels
        labels = data.loc[:, data.columns.str.startswith(label_col)]
        self._y_labels = pd.concat([labels, cv_cols], axis=1)

        return self._X_features, self._y_labels

    def get_cv_scores(
            self,
            features: pd.DataFrame,
            labels: pd.DataFrame,
            model: Union[str, BaseEstimator, keras.Model],
            cv: int = 5,
            hyperparams: Optional[dict] = None,
            prob_cutoff: float = 0.5,
            epochs: Optional[int] = None,
            batch_size: Optional[int] = None,
            callbacks: Optional = None
    ):
        """
        Obtain Cross Validation Scores for a given model. This is a custom script created to fit the .csv due to my
        molecular splits. Models will output the model, an array of CV scores and the CV mean.
        :param features: pd.DataFrame
            Features for models extracted from a pd.DataFrame.
        :param labels: pd.DataFrame
            Labels for models extracted from a pd.DataFrame.
        :param model: str
            Name of model to use.
        :param cv: int
            Number of cross-validation groups.
        :param hyperparams: Optional[dict]
            If a string is given for the model, the hyperparameters from the ..dataset/hyperparams folder will be
            pulled. Otherwise, model params will be set if a dictionary of params is given.
            the hyperparathyroidism in the appropriate file.
        :param prob_cutoff: float
            Set the probability cutoff for keras classification models. Defaults to > 0.5 to be labeled 1.
        :param epochs: Optional[int]
            Set the number of epochs to fit the model.
        :param batch_size: Optional[int]
            Set the batch size to fit the model.
        :param callbacks:
            Insert callbacks for keras model. Defaults to -> EarlyStopping(monitor='val_loss', patience=30, verbose=1)
        :return:
        """

        global model_name
        skmodel = Models()

        # get general or optimized model
        if isinstance(model, str):
            model = skmodel.get_sk_model(model)
            model_name = model.__class__.__name__
        elif isinstance(model, str) and isinstance(hyperparams, dict):
            model = skmodel.get_sk_model(model, hyperparams=hyperparams)
            model_name = model.__class__.__name__

        # if already a compiled/built model
        elif isinstance(model, BaseEstimator):
            model_name = model.__class__.__name__
        elif isinstance(model, keras.Model):
            model_name = "Keras"
            # print(model_name)  # for debugging

        # To get columns containing all CV* header
        drop_cols = labels.loc[:, labels.columns.str.startswith('CV')].columns
        y_label = labels.drop(columns=drop_cols)
        features = features.drop(columns=drop_cols)

        # get CV indexes
        cv_index = self.get_cv_split(y_label=labels, cv=cv)

        # convert y_label to numpy array for ML model
        labels = y_label.squeeze()

        # get sklearn/xgb scores
        if isinstance(model, BaseEstimator):
            results = self._sklearn_cv_score(cv_index, features, labels, model, model_name)
            return results
        # get keras scores
        elif isinstance(model, keras.Model):
            results = self._keras_cv_score(cv_index, features, labels, model, model_name, callbacks, epochs, batch_size,
                                           prob_cutoff)
            return results

    @staticmethod
    def get_cv_split(y_label: pd.DataFrame, cv: int = 5):
        """
        Obtain index for train/test split. Index is based on input CV label headers from DataFrame. This will output a
        tuple with the indices of the train and test split based on CV label headers.

        :param y_label: pd.DataFrame
            The pd.DataFrame containing the labels and cross-validation information. train_indexes and test_indexes will
            be obtained from the input label pd.DataFrame
        :param cv: int
            The number of cross-validation groups.
        :return:
        """

        splits = []

        for i in range(cv):
            header = f"CV{i}"

            train_indexes = y_label[y_label[header] == "train"].index.values.astype(int)
            test_indexes = y_label[y_label[header] == "test"].index.values.astype(int)
            splits.append((train_indexes, test_indexes))

        return splits

    @staticmethod
    def _pr_auc(expected=None, predicted=None):
        # Data to plot precision - recall curve
        precision, recall, thresholds = precision_recall_curve(expected, predicted)
        # Use AUC function to calculate the area under the curve of precision recall curve
        pr_auc_score = auc(recall, precision)
        return pr_auc_score

    @staticmethod
    def _sklearn_cv_score(cv_index, features, labels, model, model_name):
        """support function to calculate sklearn/xgb cross validation scores"""

        # List to hold cv_score
        acc_list = []
        precision_list = []
        f1_list = []
        recall_list = []
        mcc_list = []
        roc_auc_score_list = []

        acc_score = cross_val_score(
            model, features, labels, cv=cv_index, scoring="accuracy", n_jobs=-1
        )
        acc_list.append(acc_score)
        f1_test = cross_val_score(
            model, features, labels, cv=cv_index, scoring="f1", n_jobs=-1
        )
        f1_list.append(f1_test)
        recall = cross_val_score(
            model, features, labels, cv=cv_index, scoring="recall", n_jobs=-1
        )
        recall_list.append(recall)
        precision = cross_val_score(
            model, features, labels, cv=cv_index, scoring="precision", n_jobs=-1
        )
        precision_list.append(precision)
        mcc = cross_val_score(
            model, features, labels, cv=cv_index, scoring="matthews_corrcoef", n_jobs=-1
        )
        mcc_list.append(mcc)
        roc_auc = cross_val_score(
            model, features, labels, cv=cv_index, scoring="roc_auc", n_jobs=-1
        )
        roc_auc_score_list.append(roc_auc)

        # Get score averages
        results_dict = CrossVal.setup_results_dict(
            acc_list, f1_list, mcc_list, model_name, precision_list, recall_list, roc_auc_score_list
        )

        return results_dict

    @staticmethod
    def _keras_cv_score(cv_index, features, labels, model, model_name, callbacks, epochs, batch_size, prob_cutoff):
        """support function to calculate Keras CV scores"""
        # set a callback to stop training after 5 epochs to prevent overfitting.
        if callbacks is None:
            from keras._tf_keras.keras.callbacks import EarlyStopping
            callbacks = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

        # the model weights will need to be "reset" during each fold
        # store initial weights and optimizer config
        initial_weights = model.get_weights()
        optimizer_config = model.optimizer.get_config()
        optimizer_class = model.optimizer.__class__

        # todo add param for this
        loss_fn = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.Precision(name="precision")]

        # List to hold cv_score
        loss_list = []
        acc_list = []
        precision_list = []
        f1_list = []
        recall_list = []
        mcc_list = []
        roc_auc_score_list = []

        scoring_metrics = []
        cv_models = []

        fold_no = 1

        for x in cv_index:
            train_index = x[0]
            test_index = x[1]

            x_train, x_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels.iloc[train_index].values, labels.iloc[test_index].values

            # reset weights and optimizer
            model.set_weights(initial_weights)
            model.compile(optimizer=optimizer_class.from_config(optimizer_config),
                          loss=loss_fn, metrics=metrics)

            # Fit the model
            print(f'Training fold {fold_no}...')
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(x_test, y_test), callbacks=[callbacks], verbose=1)

            cv_models.append(history)

            # Evaluate the model
            predict_prob = model.predict(x_test)
            predict = (predict_prob > prob_cutoff).astype(int)  # set cutoff for 1 or 0 label with keras

            loss, accuracy, precision = model.evaluate(x_test, y_test, verbose=True)
            f1 = f1_score(y_test, predict)
            recall = recall_score(y_test, predict)
            mcc = matthews_corrcoef(y_test, predict)
            roc_auc = roc_auc_score(y_test, predict)

            # append scores
            loss_list.append(loss)
            acc_list.append(accuracy)
            precision_list.append(precision)
            f1_list.append(f1)
            recall_list.append(recall)
            mcc_list.append(mcc)
            roc_auc_score_list.append(roc_auc)

            fold_no += 1

        # get score averages
        results_dict = CrossVal.setup_results_dict(acc_list, f1_list, mcc_list, model_name, precision_list, recall_list,
                                                   roc_auc_score_list, loss_list, if_keras=True)

        return results_dict

    @staticmethod
    def setup_results_dict(acc_list, f1_list, mcc_list, model_name, precision_list, recall_list, roc_auc_score_list,
                           loss_list=None, if_keras=False):
        """support function to set up dictionary from cv scores"""
        results_dict = {}

        if if_keras:
            loss_mean = np.mean(loss_list)
            acc_mean = np.mean(acc_list)
            f1_mean = np.mean(f1_list)
            recall_mean = np.mean(recall_list)
            precision_mean = np.mean(precision_list)
            mcc_mean = np.mean(mcc_list)
            roc_auc_mean = np.mean(roc_auc_score_list)
            results_dict["Model"] = model_name
            results_dict["Loss"] = loss_list
            results_dict["Accuracy"] = acc_list
            results_dict["Precision"] = precision_list
            results_dict["Recall"] = recall_list
            results_dict["F1 Score"] = f1_list
            results_dict["MCC"] = mcc_list
            results_dict["ROC AUC"] = roc_auc_score_list
            results_dict["Loss_avg"] = loss_mean
            results_dict["Accuracy_avg"] = acc_mean
            results_dict["Precision_avg"] = precision_mean
            results_dict["Recall_avg"] = recall_mean
            results_dict["F1 Score_avg"] = f1_mean
            results_dict["MCC_avg"] = mcc_mean
            results_dict["ROC_AUC_avg"] = roc_auc_mean
        else:
            acc_mean = np.mean(acc_list)
            f1_mean = np.mean(f1_list)
            recall_mean = np.mean(recall_list)
            precision_mean = np.mean(precision_list)
            mcc_mean = np.mean(mcc_list)
            roc_auc_mean = np.mean(roc_auc_score_list)
            results_dict["Model"] = model_name
            results_dict["Accuracy"] = acc_list[0]
            results_dict["Precision"] = precision_list[0]
            results_dict["Recall"] = recall_list[0]
            results_dict["F1 Score"] = f1_list[0]
            results_dict["MCC"] = mcc_list[0]
            results_dict["ROC AUC"] = roc_auc_score_list[0]
            results_dict["Accuracy_avg"] = acc_mean
            results_dict["Precision_avg"] = precision_mean
            results_dict["Recall_avg"] = recall_mean
            results_dict["F1 Score_avg"] = f1_mean
            results_dict["MCC_avg"] = mcc_mean
            results_dict["ROC_AUC_avg"] = roc_auc_mean

        return results_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()
