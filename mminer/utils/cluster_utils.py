import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import AgglomerativeClustering
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from mminer.utils.fingerprint_misc import smiles_to_fp
from typing import List, Optional, Union

__all__ = ["ButinaClusters", "UmapClusters"]

"""
Additional splitting methods not available in splito. Methods are written to fit sklearn and splito format where 
possible.
"""


# Custom splitter for Butina clustering
# Code adapted from here: https://github.com/PatWalters/useful_rdkit_utils
class ButinaClusters(GroupShuffleSplit):
    def __init__(
            self,
            smiles: Union[List[str], np.array] = None,
            cutoff: float = 0.65,
            n_splits: int = 5,
            method: str = 'morgan',
            radius: int = 2,
            nbits: int = 2048,
            bitvector: bool = True,
            test_size: Optional[Union[float, int]] = None,
            train_size: Optional[Union[float, int]] = None,
            random_state: Optional[Union[int, RandomState]] = None,
    ):
        """
        Initialize the ButinaClusters class
        :param smiles: Union[List[str], np.array]
            A list of smiles string for processing
        :param cutoff: float
            Set the tanimoto similarity cut off for clustering.
        :param n_splits: int
            Set the number of splits
        :param method: str
            The type of fingerprint method to use. Defaults to morgan.
        :param radius: int
            Set the radius for circular fingerprints.
        :param nbits: int
            The number of bits to use. Defaults to 2048.
        :param bitvector: bool
            If morgan fingerprint, must be an explicit bit vector. Will default to true for butina clustering.
        :param test_size: Optional[Union[float, int]]
            Set the test side for splits.
        :param train_size: Optional[Union[float, int]]
            Set the train size for splits.
        :param random_state: Optional[Union[int, RandomState]]
            Set the seed for repeatability.
        """
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.threshold = cutoff
        self._smiles = smiles
        self.radius = radius
        self.method = method
        self.nbits = nbits
        self.bitvector = bitvector

    def _iter_indices(
            self,
            X: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
    ):
        """
        Generate (train, test) indices for ButinaClusters. Follows splito format where possible.
        Parameters:
            X: Input features (list of smiles strings to be converted into fingerprints).
            y: Optional target values.
            groups: Optional group identifiers.

        Yields:
            (train_idx, test_idx): Train-test index pairs.
        """
        # check smiles strings
        requires_smiles = X is None or not all(isinstance(x, str) for x in X)
        if self._smiles is None and requires_smiles:
            raise ValueError(
                "If the input is not a list of SMILES, you need to provide the SMILES to the constructor."
            )

        smiles = self._smiles if requires_smiles else X

        fp_list = [smiles_to_fp(x, method=self.method, radius=self.radius, nbits=self.nbits, bitvector=self.bitvector)
                   for x in smiles]

        # Generate Tanimoto similarity matrix
        dist_matrix = []
        for i in range(len(fp_list)):
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            dist_matrix.extend([1 - sim for sim in sims])  # Distance = 1 - similarity

        # Butina clustering
        clusters = Butina.ClusterData(
            dist_matrix, len(fp_list), self.threshold, isDistData=True
        )

        # Assign cluster ID to group labels
        groups = np.zeros(len(fp_list), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                groups[idx] = cluster_id

        # Generate splits based on clusters using GroupShuffleSplit
        yield from super()._iter_indices(X, y, groups)

# Custom splitter for Umap clustering
# Code adapted from here: https://github.com/PatWalters/useful_rdkit_utils
class UmapClusters(GroupShuffleSplit):
    def __init__(
            self,
            smiles: Union[List[str], np.array] = None,
            n_splits: int = 5,
            n_clusters: int = 7,
            method: str = 'morgan',
            radius: int = 2,
            nbits: int = 2048,
            bitvector: bool = True,
            test_size: Optional[Union[float, int]] = None,
            train_size: Optional[Union[float, int]] = None,
            random_state: Optional[Union[int, RandomState]] = None
    ):
        """
        Initialize the ButinaClusters class
        :param smiles: Union[List[str], np.array]
            A list of smiles string for processing
        :param n_splits: int
            Set the number of splits
        :param method: str
            The type of fingerprint method to use. Defaults to morgan.
        :param radius: int
            Set the radius for circular fingerprints.
        :param nbits: int
            The number of bits to use. Defaults to 2048.
        :param bitvector: bool
            If morgan fingerprint, must be an explicit bit vector. Will default to true for butina clustering.
        :param test_size: Optional[Union[float, int]]
            Set the test side for splits.
        :param train_size: Optional[Union[float, int]]
            Set the train size for splits.
        :param random_state: Optional[Union[int, RandomState]]
            Set the seed for repeatability.
        """
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._smiles = smiles
        self.radius = radius
        self.method = method
        self.nbits = nbits
        self.bitvector = bitvector
        self.n_clusters = n_clusters
        self.n_splits = n_splits

    def _iter_indices(
            self,
            X: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            n_clusters: int = 10,  # Number of clusters for UMAP
    ):
        """
        Generate (train, test) indices for UMAPClusters.
        Parameters:
            X: Input features (list of SMILES strings to be converted into fingerprints).
            y: Optional target values.
            groups: Optional group identifiers.

        Yields:
            (train_idx, test_idx): Train-test index pairs.
        """
        # Check for SMILES
        requires_smiles = X is None or not all(isinstance(x, str) for x in X)
        if self._smiles is None and requires_smiles:
            raise ValueError(
                "If the input is not a list of SMILES, you need to provide the SMILES to the constructor."
            )

        smiles = self._smiles if requires_smiles else X

        # Convert SMILES to fingerprints
        fp_list = [smiles_to_fp(x, method=self.method, radius=self.radius, nbits=self.nbits, bitvector=self.bitvector)
                   for x in smiles]

        # Perform Agglomerative Clustering on the fingerprints
        ac = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = ac.fit_predict(np.stack(fp_list))

        # Assign cluster labels as groups for splitting
        groups = cluster_labels

        # Generate splits based on clusters using GroupShuffleSplit
        yield from super()._iter_indices(X, y, groups)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
