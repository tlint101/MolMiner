"""
Scripts to fit and generate predictions from models.
"""

from mminer.models.skmodels import Models
from mminer.utils.cluster_utils import ButinaClusters, UmapClusters
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import splito

__all__ = ["ml_results"]


class ml_results:
    def __init__(self, model):
        self.model = Models.get_sk_model(model)

    def fit_model(self, X_train, y_train):
        """
        Fit the model according to the training data.
        :param X_train: pd.DataFrame
            Training features from the input DataFrame.
        :param y_train: pd.Series
            Training labels from the input Dataframe.
        :return:
        """
        model = self.model.fit(X_train, y_train)
        return model

    def predict(self, predict_list):
        """
        Provide prediction result from the input features.

        :param predict_list: pd.DataFrame
            Provide features for Prediction results for the model.
        :return:
        """
        self._y_predict = self.model.predict(predict_list)
        return self._y_predict


class Splitting_utils:
    def __init__(self, model):
        pass

    @staticmethod
    def get_split_mapping(smi_col, test_size, n_splits, cutoff, n_clusters, seed):
        # todo add kfold split?
        mapping = {
            "Random": lambda: ShuffleSplit(test_size=test_size, random_state=seed, n_splits=n_splits),

            "StratifiedShuffleSplit": lambda: StratifiedShuffleSplit(test_size=test_size, random_state=seed,
                                                                     n_splits=n_splits),
            "Scaffold": lambda: splito.ScaffoldSplit(
                smi_col, test_size=test_size, random_state=seed, n_splits=n_splits, n_jobs=-1),

            "Perimeter": lambda: splito.PerimeterSplit(
                test_size=test_size, random_state=seed, n_splits=n_splits, n_jobs=-1),

            "MaxDissimilarity": lambda: splito.MaxDissimilaritySplit(
                test_size=test_size, random_state=seed, n_splits=n_splits, n_jobs=-1),

            "MolecularMinxMaxSplit": lambda: splito.MolecularMinMaxSplit(
                smiles=smi_col, test_size=test_size, random_state=seed, n_splits=n_splits),

            "MolecularWeight": lambda: splito.MolecularWeightSplit(
                smiles=smi_col, test_size=test_size, random_state=seed, n_splits=n_splits),

            "KMeansSplit": lambda: splito.KMeansSplit(test_size=test_size, random_state=seed),

            "Butina": lambda: ButinaClusters(
                smiles=smi_col, cutoff=cutoff, n_splits=n_splits, test_size=test_size, random_state=seed),

            "Umap": lambda: UmapClusters(
                smiles=smi_col, n_splits=n_splits, test_size=test_size, n_clusters=n_clusters, random_state=seed)
        }

        return mapping


if __name__ == "__main__":
    import doctest

    doctest.testmod()
