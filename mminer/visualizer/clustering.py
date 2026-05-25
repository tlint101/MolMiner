"""
Scripts to cluster molecules
"""
import pandas as pd
import numpy as np
import seaborn as sns
import datamol as dm
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.ML.Cluster import Butina
from sklearn.cluster import KMeans, MiniBatchKMeans
from mminer.features.fingerprint import Fingerprint


__all__ = ["Cluster"]


class Cluster:
    def __init__(
            self, data: pd.DataFrame = None, smi_col: str = None):
        """
        InitializeCluster class
        :param data: pd.DataFrame
            Input DataFrame containing at least a column of smiles string.
        :param smi_col: str
            Name of column containing smiles string.
        """
        self.data = data
        self.smi_col = smi_col

    def show(self, rows=None):
        """
        show DataFrame

        :param rows: Int
            Indicate the number of rows to display. If none, automatically show 5.
        :return: DataFrame
        """
        returned_df = self.data

        if rows is None:
            # print("rows is none") # for troubleshooting
            return returned_df.head()
        elif isinstance(rows, int):
            # print("rows are given!") # for troubleshooting
            return returned_df.head(rows)

    def butina(
            self,
            data: pd.DataFrame = None,
            smi_col: str = None,
            cutoff: float = 0.2,
            method: str = "morgan",
            radius: int = 2,
            nbits: int = 2048,
            output: str = "all",
            verbose: bool = False,
    ):
        """
        Cluster molecules. Input data should be molecules as DataFrame. Has parameters that can be changed. If none,
        will only include default params from initialization. Function will calculate fingerprints, similarity of every
        molecule pairs, create a distance matrix for pairwise similarity value, and assign a cluster id to each
        molecule.
        :param data: pd.DataFrame
            Input dataframe. Default is self.
        :param smi_col: str
            String of single column containing smiles string. Default is self.smi_col.
        :param cutoff: float
            Cutoff for clustering group.Default is 0.2.
        :param method: str
            Molecular fingerprint method. Default is 'morgan'.
        :param radius: int
            Radius for circular fingerprints. Default is 2.
        :param nbits: int
            Number of bits for circular fingerprints. Default is 2048.
        :param verbose: bool
            Print clustering report. Default is False.
        :param output:
        :return:
        """

        if data is None or smi_col is None or cutoff is None:
            if data is None:
                data = self.data
            if smi_col is None:
                smi_col = self.smi_col

        # Generate fingerprint
        fp_generator = Fingerprint()
        tqdm.pandas(desc="Generating fingerprint")
        data["fp"] = data[smi_col].progress_apply(
            lambda smi: fp_generator.smi_to_fp(
                smi=smi, method=method, radius=radius, nbits=nbits, bitvector=True
            )
        )

        # convert fp into a list for processing
        fp_list = data["fp"].values.tolist()
        # return fp_list

        """Cluster molecules"""
        distance_matrix = []
        fp_list_len = len(fp_list)
        # Compare the current fingerprint against all in list and output distance matrix
        for i in tqdm(range(1, fp_list_len), desc="Generating Distance Matrix"):
            similarities = Chem.DataStructs.BulkTanimotoSimilarity(
                fp_list[i], fp_list[:i]
            )
            distance_matrix.extend([1 - x for x in similarities])

        # Cluster molecules using Butina
        mol_clusters = Butina.ClusterData(
            distance_matrix, fp_list_len, cutoff, isDistData=True
        )

        # Calculating a list of centroid idx
        centroid_list = []
        for tup in mol_clusters:
            centroid_idx = tup[0]
            centroid_list.append(centroid_idx)

        mol_clusters = sorted(mol_clusters, key=len, reverse=True)
        cluster_id_list = [0] * fp_list_len
        for idx, cluster in enumerate(mol_clusters, 1):
            # print(cluster) # for debugging
            for member in cluster:
                cluster_id_list[member] = idx

        if verbose:
            # Give a short report about the numbers of clusters and their sizes
            num_clust_g1 = sum(1 for c in mol_clusters if len(c) == 1)
            num_clust_g5 = sum(1 for c in mol_clusters if len(c) > 5)
            num_clust_g25 = sum(1 for c in mol_clusters if len(c) > 25)
            num_clust_g100 = sum(1 for c in mol_clusters if len(c) > 100)

            print("total # clusters: ", len(mol_clusters))
            print("# clusters with only 1 compound: ", num_clust_g1)
            print("# clusters with >5 compounds: ", num_clust_g5)
            print("# clusters with >25 compounds: ", num_clust_g25)
            print("# clusters with >100 compounds: ", num_clust_g100)

        # conditional to output centroid only, or min/max molecule for each cluster
        if output == "all":
            self.data["cluster"] = cluster_id_list
            self.data["centroid"] = 0
            self.data.loc[centroid_list, "centroid"] = 1  # column with centroid label
            # variable for output
            output_df = self.data
            return output_df

        elif output == "centroid":
            self.data["cluster"] = cluster_id_list
            self.data["centroid"] = 0
            self.data.loc[centroid_list, "centroid"] = 1
            # variable for output
            output_df = self.data.iloc[centroid_list]
            return output_df

        elif output == "max":
            self.data["cluster"] = cluster_id_list
            self.data["ExactMolWt"] = self._get_mw()

            # Get index of row with maximum weight for each cluster
            max_weight_indices = self.data.groupby("cluster")["ExactMolWt"].idxmax()

            output_df = self.data.loc[max_weight_indices]
            return output_df

        elif output == "min":
            self.data["cluster"] = cluster_id_list
            self.data["ExactMolWt"] = self._get_mw()

            # Get index of row with minimum weight for each cluster
            min_weight_indices = self.data.groupby("cluster")["ExactMolWt"].idxmin()

            output_df = self.data.loc[min_weight_indices]
            return output_df

        else:
            raise ValueError(
                "Output must be either 'all', 'centroid' or 'max' or 'min'"
            )

    def kmeans(self,
               data: pd.DataFrame = None,
               clusters: int = 3,
               method: str = 'kmeans',
               batch_size: int = 100,
               smi_col: str = None,
               fp_method: str = "morgan",
               radius: int = 2,
               nbits: int = 2048,
               verbose: int = 0,
               seed: int = 42
               ):
        """
        Cluster molecules. Input data should be molecules as DataFrame. Has parameters that can be changed. If none,
        will only include default params from initialization. Function will calculate fingerprints, similarity of every
        molecule pairs, create a distance matrix for pairwise similarity value, and assign a cluster id to each
        molecule.
        :param data: pd.DataFrame
            Input dataframe. Default is self.
        :param clusters: int
            Cutoff for clustering group.Default is 0.2 from self.cutoff
        :param method: str
            Method for kmeans split. Currently two are possible - 'kmeans' and 'minibatch'. Use minibatch for larger
            datasets and speed. Use of minibatch also requires modification to batch_size param.
        :param batch_size: int
            Set size of minibatches. Only used for 'minibatch' method.
        :param smi_col: str
            String of single column containing smiles string. Default is self.smi_col.
        :param fp_method: str
            Molecular fingerprint method. Default is 'morgan'.
        :param radius: int
            Radius for circular fingerprints. Default is 2.
        :param nbits: int
            Number of bits for circular fingerprints. Default is 2048.
        :param verbose: int
            Add verbosity level. Default is 0 (No feedback)
        :param seed: int
            Seed for random number generator. Default is 42.
        :return:
        """

        if data is None or smi_col is None or clusters is None:
            if data is None:
                data = self.data
            if smi_col is None:
                smi_col = self.smi_col
            if clusters is None:
                clusters = self.clusters

        # Generate fingerprint
        fp_generator = Fingerprint()
        tqdm.pandas(desc="Generating fingerprint")
        data["fp"] = data[smi_col].progress_apply(
            lambda smi: fp_generator.smi_to_fp(smi=smi, method=fp_method, radius=radius, nbits=nbits, bitvector=True))

        # convert fp into a list for processing
        fp_list = np.array(data["fp"].values.tolist())
        # return fp_list

        """kmeans clustering method"""
        if method == "kmeans":
            kmeans = KMeans(n_clusters=clusters, verbose=verbose, random_state=seed)
        elif method == "minibatch":
            kmeans = MiniBatchKMeans(n_clusters=clusters, batch_size=batch_size, verbose=verbose, random_state=seed)
        else:
            raise ValueError("Method must be 'kmeans' or 'minibatch'!!!!")

        kmeans.fit(fp_list)

        # get labels
        cluster_labels = kmeans.labels_
        data['cluster'] = cluster_labels

        # get centroid
        data['centroid'] = 0
        for cluster in range(clusters):
            centroid_index = np.argmin(np.linalg.norm(fp_list - kmeans.cluster_centers_[cluster], axis=1))
            data.loc[centroid_index, 'centroid'] = 1
        return data

    def viz_cluster_size(
            self,
            data: pd.DataFrame = None,
            return_df: bool = False,
            **kwargs,
    ):
        """
        Visualize cluster size with a histogram. Method requires a pd.DataFrame with clusters already calculated using
        methods from the Cluster() class. Default x value will search for a column titled "cluster". If input
        DataFrame is None, then the Dataframe used will be from the class initializer.

        :param data: pd.DataFrame
            Input dataframe. Default is self.
        :param return_df:
            Optional. Return DataFrame of molecules with cluster label.
        :param kwargs: Seaborn keywords arguments
        :return: matplotlib.axes.Axes
        """
        # run calculations if data is None
        if data is None:
            data = self.data

        # if 'x' in kwargs is None, set to "cluster"
        if "x" not in kwargs:
            x = "cluster"
            # histplot with specific params for plotting clusters
            fig = sns.histplot(data=data, x=x, discrete=True, stat="count", **kwargs)
        else:
            fig = sns.histplot(data=data, discrete=True, stat="count", **kwargs)

        if return_df:
            return fig, data
        else:
            return fig

    def output_cluster_group(
            self, data: pd.DataFrame = None, cluster: int = None, cluster_col: str = None
    ):
        """
        Output cluster group from dataframe. This is used to quickly check the molecules in a given cluster.
        :param data: pd.DataFrame
            Input dataframe with cluster results. Default is self.data.
        :param cluster: int
            Cluster Number.
        :param cluster_col: str
            Set column containing cluster number. If None, will default to "cluster".
        :return:
        """

        if data is None:
            if data is None:
                data = self.data
        if cluster_col is None:
            cluster_col = "cluster"
        if cluster is None:
            raise ValueError("Specify Cluster Number")

        output = data[data[cluster_col] == cluster]

        return output

    def diverse_mols(self,
                     data: pd.DataFrame = None,
                     smi_col: str = None,
                     npick: int = 30,
                     to_image: bool = False,
                     mol_name: str = None,
                     n_cols: int = 4,
                     mol_size=(150, 100),
                     seed: int = 42):
        """
        Input a list of molecules and get a list of diverse molecules on output. Output should be in pd.DataFrame
        format.
        :param data: pd.DataFrame
            Input query molecules
        :param smi_col: str
            Indicate column name containing molecule smiles string.
        :param npick: int
            The number of diverse molecules to output. Default is 30.
        :param to_image: bool
            If true, return image of diverse molecules in a grid.
        :param mol_name: str
            Optional. Param only functions if to_image is True. Indicate column name containing molecule name.
        :param n_cols: int
            Only functions if to_image is True. Indicate the number of columns when drawing molecules in a grid.
        :param mol_size: tuple
            Set the size of the molecules in the grid. Param is only used if the to_image param is set to True. Default
            is (150, 100).
        :param seed: int
            Set the reproducibility seed. Default is 42.
        :return:
        """
        # If None, set to self versions
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smi_col

        # Convert DataFrame column into a list
        smi_list = data[smi_col].tolist()

        # Get diverse picks
        indices, picks = dm.pick_diverse(smi_list, npick=npick, seed=seed, n_jobs=-1)

        # Optional: draw diverse set in a grid.
        if to_image is True:
            img = dm.to_image(picks, legends=mol_name, mol_size=mol_size, n_cols=n_cols, align=True)
            return img
        else:
            return picks

    def _get_mw(self):
        """Support function to get molecular weight of molecules"""
        global wt
        wt = []
        for mol in self.data[self.smi_col]:
            calculated_wt = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(mol))
            wt.append(round(calculated_wt, 3))
        return wt

    # todo add kmedoids when sklearn_extras has been updated
    # def kmedoids(self,
    #              data: pd.DataFrame = None,
    #              clusters: int = 3,
    #              metric: str = 'euclidean',
    #              smi_col: str = None,
    #              fp_method: str = "morgan",
    #              radius: int = 2,
    #              nbits: int = 2048,
    #              seed: int = 42
    #              ):
    #     """
    #     Cluster molecules. Input data should be molecules as DataFrame. Has parameters that can be changed. If none,
    #     will only include default params from initialization. Function will calculate fingerprints, similarity of every
    #     molecule pairs, create a distance matrix for pairwise similarity value, and assign a cluster id to each
    #     molecule.
    #     :param data: pd.DataFrame
    #         Input dataframe. Default is self.
    #     :param clusters: int
    #         Cutoff for clustering group.Default is 0.2 from self.cutoff
    #     :param metric: str
    #         Set which distance to use. Will default to euclidean.
    #     :param smi_col: str
    #         String of single column containing smiles string. Default is self.smi_col.
    #     :param fp_method: str
    #         Molecular fingerprint method. Default is 'morgan'.
    #     :param radius: int
    #         Radius for circular fingerprints. Default is 2.
    #     :param nbits: int
    #         Number of bits for circular fingerprints. Default is 2048.
    #     :param seed: int
    #         Seed for random number generator. Default is 42.
    #     :return:
    #     """
    #
    #     if data is None or smi_col is None or clusters is None:
    #         if data is None:
    #             data = self.data
    #         if smi_col is None:
    #             smi_col = self.smi_col
    #         if clusters is None:
    #             clusters = self.clusters
    #
    #     # Generate fingerprint
    #     tqdm.pandas(desc="Generating fingerprint")
    #     data["fp"] = data[smi_col].progress_apply(
    #         lambda smi: smi_to_fp(
    #             smi=smi, method=fp_method, radius=radius, nbits=nbits, bitvector=True
    #         )
    #     )
    #
    #     # convert fp into a list for processing
    #     fp_list = np.array(data["fp"].values.tolist())
    #     # return fp_list
    #
    #     """kmeans clustering"""
    #     kmedoids = KMedoids(n_clusters=clusters, metric=metric, random_state=seed)
    #     kmedoids.fit(fp_list)
    #
    #     # get labels
    #     cluster_labels = kmedoids.labels_
    #     data['cluster'] = cluster_labels
    #
    #     data['centroid'] = 0
    #     for cluster in range(clusters):
    #         centroid_index = np.argmin(np.linalg.norm(fp_list - kmedoids.cluster_centers_[cluster], axis=1))
    #         data.loc[centroid_index, 'centroid'] = 1
    #
    #     return data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
