"""
Script to generate representations of query chemical space.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import splito
from tqdm import tqdm
import datamol as dm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, DataStructs, PandasTools
from mminer.features.fingerprint import Fingerprint
from mminer.features.build import Score
from typing import Union

__all__ = ["MolecularFragments", "Similarity", "reduce_dimension", "LibrarySpace", "FragmentSpace"]


def training_space(features_array=None, features_smi=None, test_size=0.2, random_state=42, n_splits=10,
                   **kwargs):
    """
    Split data into training and test set. Utilizes different ways to parse datasets. Each parsing is then scored. The
    training and test sets for each parsing is done to provide a uniform distribution of the features. The score is
    calculated by measuring the distance between the test-to-train distribution and the deployment-to-train distance
    distribution

    :param features_array: An array of features. For this, use the molecular fingerprints of a compound. The fingerprint
    array is a "column" from a larger DataFrame.
    :param features_smi: A list of smiles string. This can be taken from a DataFrame using "df['Std_Smiles'].values"
    :param test_size:
    :param random_state: For reproducibility. This is set to 42 by default.
    :param n_splits:
    :param kwargs: additional arguments to pass to as splitters. This should be written as a dictionary, for example:
    'Random': ShuffleSplit(random_state=random_state),  # same as train_test_split

    :return: DataFrame containing scoring results.
    """
    splitters = {
        "Random": ShuffleSplit(
            test_size=test_size, random_state=random_state, n_splits=n_splits
        ),  # same as train_test_split
        "Scaffold": splito.ScaffoldSplit(
            features_smi,
            test_size=test_size,
            random_state=random_state,
            n_splits=n_splits,
        ),
        "Perimeter": splito.PerimeterSplit(
            test_size=test_size, random_state=random_state, n_splits=n_splits
        ),
        "MaxDissimilarity": splito.MaxDissimilaritySplit(
            test_size=test_size, random_state=random_state, n_splits=n_splits
        ),
        "MolecularMinxMaxSplit": splito.MolecularMinMaxSplit(
            smiles=features_smi,
            test_size=test_size,
            random_state=random_state,
            n_splits=n_splits,
        ),
        "MolecularWeight": splito.MolecularWeightSplit(
            smiles=features_smi,
            test_size=test_size,
            random_state=random_state,
            n_splits=n_splits,
        ),
        "KMeansSplit": splito.KMeansSplit(
            test_size=test_size, random_state=random_state, n_splits=n_splits
        ),
        **kwargs,
    }

    splitter = splito.MOODSplitter(splitters)

    split_result = splitter.fit(
        X=np.stack(features_array), X_deployment=np.stack(features_array)
    )

    return split_result


def reduce_dimension(method: str = None, data: pd.DataFrame = None, fp_col: Union[str, np.ndarray] = None,
                     drop_fp_col: bool = False, n_components: int = 2, seed: int = 42, **kwargs):
    """
    Visualize chemical space. Takes in a DataFrame and a column containing molecular fingerprints. Compound smile
    strings will be converted into a fingerprint adn then dimensions will be reduced using pca, tsne, or umap. Output
    will be a DataFrame containing dimension result.
    :param method: str
        Denote the type of dimension reduction to use. Options include pca, tsne, or umap.
    :param data: pd.DataFrame
        Input data as a pd.DataFrame.
    :param fp_col: str or np.Array
        String of suffix for columns containing fingerprint or an np.array of fingerprint columns. If string, columns
        will be sorted and filtered using regex.
    :param drop_fp_col: bool
         Determine whether to drop the fingerprint columns or retain them.
    :param n_components: int
        Set the number of dimensions to reduce to. Default is 2.
    :param seed: int
        Set the reproducibility of the dimension reduction.
    :param kwargs: Additional keyword arguments. For PCA and TSNE, see https://scikit-learn.org. For UMAP,
    see https://umap-learn.readthedocs.io/en/latest/
    :return:
    """

    # Check format of the fp column. At minimum, it must be in np.array. Can try tweaking Fingerprint.smi_to_fp()
    global fp_array
    try:
        # if the Fingerprint.expand_fp function is not used:
        if isinstance(data, pd.DataFrame) and isinstance(data[fp_col], pd.Series):
            fp_generator = Fingerprint()
            data = fp_generator.expand_fp(data, fp_col=fp_col)
            fp_columns = data.filter(regex=fp_col)
            fp_array = fp_columns.to_numpy()
    except KeyError:
        # if Fingerprint.expand_fp is used, then run this
        if isinstance(fp_col, str):
            fp_columns = data.filter(regex=fp_col)
            fp_array = fp_columns.to_numpy()

    if fp_array is None:
        raise ValueError("Issue with fp_array!")

    # Dimension reduction
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=seed, **kwargs)
        result = pca.fit_transform(fp_array)
        # Add the PCA results to the DataFrame
        for i in tqdm(range(n_components), desc='Reducing dimension with PCA'):
            data[f"PC{i + 1}"] = result[:, i]
    elif method == "tsne":
        tsne = TSNE(n_components=n_components, random_state=seed, **kwargs)  # todo add perplexitiy?
        result = tsne.fit_transform(fp_array)
        for i in tqdm(range(n_components), desc='Reducing dimension with T-SNE'):
            data[f"tsne{i + 1}"] = result[:, i]
    elif method == "umap":
        fit = umap.UMAP(random_state=seed, **kwargs)
        result = fit.fit_transform(fp_array)
        for i in tqdm(range(n_components), desc='Reducing dimension with UMAP'):
            data[f"umap{i + 1}"] = result[:, i]
    else:
        raise ValueError(
            "Invalid method. Supported methods are 'pca', 'tsne', or 'umap'."
        )

    # bool to drop fp columns
    if drop_fp_col:
        cols_to_drop = data.filter(regex=fp_col)
        data = data.drop(columns=cols_to_drop)

    return data


"""For easier fragment counts of kinome"""


class FragmentSpace:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def space(self, data: pd.DataFrame = None, smi_col: str = None, target_col: str = 'target',
              total_col: str = 'total_count'):
        """
        This function condenses many functions. A matrix will be generated by using the pivot_table and will be based on
        the fragment target and total counts. Molecules will then be scaled using Normalizer() and the matrix columns
        will be sorted to have rows with non-zero values in the columns to come first. This is for easier use for final
        heatmap plot.
        :param data:
        :param smi_col:
        :return:
        """
        if data is None:
            data = self.data

        # Generate a matrix based on general fragment dataset
        matrix = data.pivot_table(index=smi_col, columns=target_col, values=total_col, fill_value=0)

        # Flatten and drop index
        matrix.reset_index(inplace=True)
        matrix.columns.name = None
        matrix.set_index(smi_col, inplace=True)

        # scale fragments
        scaler = Normalizer()
        scaler.fit(matrix)
        scaled_values = scaler.transform(matrix)

        scaled_matrix = pd.DataFrame(scaled_values, columns=matrix.columns, index=matrix.index)

        kinome_cols = ['AGC', 'Atypical', 'CAMK', 'CK1', 'CMGC', 'Other', 'STE', 'TK', 'TKL']

        # Iteratively apply the sorting function to each column
        for col in kinome_cols:
            scaled_matrix = _sort_by_column(scaled_matrix, col)

        return scaled_matrix

    def extract_fragment(self, data=None, target_col: str = None, cutoff=0.1, threshold=0.1):
        """
        Extract fragments from the clustermap.
        :param data: pd.DataFrame
            Load the dataframe containing the fragment clustermpa.
        :param target_col: str
            Target column name.
        :param cutoff: float
            Set the cutoff for the target column. Values in the column that are less than the input value will be removed.
        :param threshold: float
            Set the threshold for rows in the other columns. Rows where columns are greater than the threshold value will be removed.
        :return:
        """
        if data is None:
            data = self.data

        # get all columns minus target_col, separate string columns
        other_cols = data.select_dtypes(include=['number'])

        # filter on numeric cols
        other_cols = other_cols.columns[other_cols.columns != target_col]
        data = data[data[target_col] >= cutoff]  # target_col cutoff
        data = data[data[other_cols].le(threshold).all(axis=1)]  # threshold cutoff for other cols

        return data


def _sort_by_column(df, col):
    # support function for fragment space Function to sort DataFrame rows based on a specific column
    # Sort the dataframe such that rows with non-zero values in `col` come first
    df = df.iloc[df[col].to_numpy().argsort(kind='stable')[::-1]]
    return df


""" 
Scripts to visualize molecular fingerprints
"""


class MolecularFragments:
    def __init__(self, smiles: str = None):
        self.smiles = smiles

    def visualize_fp(
            self,
            smiles: str = None,
            bit_query: int or list = None,
            molsPerRow: int = 1,
            fp_type: str = "morgan",
            **kwargs,
    ):
        """
        Visualize the fingerprint bits for a given molecule. Currently only Morgan and RDKIT Fingerprint are available
        options!

        :param smiles: str
             The smiles string of query molecule
        :param bit_query: int or list
             Can be a lone int or a list of ints. This will dictate the number of molecules to draw per row. Higher
             number will indicate a grid of drawn fingerprint bits.
        :param molsPerRow: int
            Number of molecules to be drawn per row. Default is 1 and assumes only 1 fragment will be queried.
        :param fp_type: str
            Name of fingerprint method. Currently only morgan and rdkit are available.
        :param kwargs:
            Additional options for radius, nBits, and maxPath can be given here. By default, radius=2, nBits=2048, and
            maxPath=7.
        :return:
        """
        global fig
        if smiles is None:
            smiles = self.smiles

        # Extract optional arguments for fingerprint calculation
        radius = kwargs.get("radius", 2)
        nBits = kwargs.get("nBits", 2048)
        maxPath = kwargs.get("maxPath", 7)

        # Convert smiles to rdkit mol
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise ValueError("SMILES input is not a valid SMILES string")

        if fp_type == "morgan":
            bit_info = {}  # dictionary to hold information
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol=mol, radius=radius, nBits=nBits, bitInfo=bit_info
            )
        elif fp_type == "rdkit":
            bit_info = {}
            fp = Chem.RDKFingerprint(mol, maxPath=maxPath, bitInfo=bit_info)
        else:
            raise ValueError("fp_type must be either 'morgan' or 'rdkit'!!)")

        # Convert specific_bit to a list if it's not already a list
        if bit_query is None:
            raise ValueError("Need a bit query for molecular fingerprint!")
        elif not isinstance(bit_query, list):
            bit_query = [bit_query]
        elif isinstance(bit_query, list):
            pass
        else:
            raise ValueError("Maybe an issue with the bit_query?")

        # Draw fingerprint bit. The tuple contains molecule, bit query, and bit information.
        try:
            if fp_type == "morgan":
                tuple_info = [(mol, bit, bit_info) for bit in bit_query]
                fig = Draw.DrawMorganBits(
                    tuple_info,
                    molsPerRow=molsPerRow,
                    legends=[str(bit) for bit in bit_query],
                )

            elif fp_type == "rdkit":
                tuple_info = [(mol, bit, bit_info) for bit in bit_query]
                fig = Draw.DrawRDKitBits(
                    tuple_info,
                    molsPerRow=molsPerRow,
                    legends=[str(bit) for bit in bit_query],
                )

        except:
            raise ValueError("Issue with drawing molecule! Check bit_query and nBits!")

        return fig

    def show_bit_info(
            self,
            smiles: str = None,
            bit_query: int = None,
            fp_type: str = "morgan",
            **kwargs,
    ):
        """
        Similar to visualize_fp(), but instead of drawing fragment, only the relevant bit information will be printed.
        This is useful to identify specific bit vector to draw. The results will be shown as a dictionary.
        :param smiles: str
             The smiles string of query molecule
        :param bit_query: int or list
             Can be a lone int or a list of ints. This will dictate the number of molecules to draw per row. Higher
             number will indicate a grid of drawn fingerprint bits.
        :param fp_type: str
            Name of fingerprint method. Currently only morgan and rdkit are available.
        :param kwargs:
            Additional options for radius, nBits, and maxPath can be given here. By default, radius=2, nBits=2048, and
            maxPath=7.
        :return:
        """
        if smiles is None:
            smiles = self.smiles

        # Extract optional arguments for fingerprint calculation
        radius = kwargs.get("radius", 2)
        nBits = kwargs.get("nBits", 2048)
        maxPath = kwargs.get("maxPath", 7)

        # Convert smiles to rdkit mol
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise ValueError("SMILES input is not a valid SMILES string!")

        # Generate molecular fingerprint
        if fp_type == "morgan":
            bit_info = {}  # dictionary to hold information
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius, nBits, bitInfo=bit_info
            )
        elif fp_type == "rdkit":
            bit_info = {}
            fp = Chem.RDKFingerprint(mol, maxPath=maxPath, bitInfo=bit_info)
        else:
            raise ValueError(
                "fp_type must be either 'morgan' or 'rdkit' fingerprints!)"
            )

        return bit_info

    def heatmap(self):
        """
        Generate a heatmap from a dataframe of fragments.
        :return:
        """

        pass


""" 
Scripts to generate similarity matrix
"""


class Similarity:
    def __init__(self, data: pd.DataFrame = None, smi_col: str = None, name_col: str = None):
        """
        Initialize the Similarity() class. The parameters are optional.
        :param data: pd.DataFrame
            DataFrame containing the compound name and smiles string.
        :param smi_col: str
            The DataFrame column containing the smile string.
        :param name_col: str
            The DataFrame column containing the name string.
        """

        # generate instance variables
        self.data = data
        self.smi_col = smi_col
        self.name_col = name_col
        self.matrix = None

    def get_diverse(self, data: pd.DataFrame = None, smi_col: str = None, npick: int = 30, seed: int = 42):
        """
        Get a diverse set of molecules.

        :param data: pd.DataFrame
            Input DataTable of molecules.
        :param smi_col:  str
            Column designating the smiles string.
        :param npick: Int
            Set the number of diverse molecules to return.
        :param seed: int
            Set the seed for repeatability.
        :return:
        """

        # match to instance variables
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smi_col

        diverse_index, diverse_smi = dm.pick_diverse(mols=data[smi_col], npick=npick, seed=seed)

        diverse_df = data.iloc[diverse_index].reset_index(drop=True)

        return pd.DataFrame(diverse_df)

    def similarity_matrix(self, data: pd.DataFrame = None, name_col: str = None, smi_col: str = None,
                          radius: int = 2, nBits: int = 2048):
        """
        Get a pd.DataFrame of the similarity matrix. Results are not yet clustered. By default, the similarity matrix 
        will convert the molecules into circular (Morgan) fingerprints.  

        :param data: pd.DataFrame
             Input DataTable of molecules.
        :param name_col: str
            Column designating the compound names.
        :param smi_col: str
            Column designating the smiles string.
        :param radius: int
            Set the radius.  
        :param nBits: int
            Set the number of bits for the molecular fingerprint. 
        :return:
        """

        # match to instance variables
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smi_col
        if name_col is None:
            name_col = self.name_col

        # Create list for Tanimoto scores
        Tanimoto = []

        # Add ROMol to DataFrame
        PandasTools.AddMoleculeColumnToFrame(data, smilesCol=smi_col)

        for compound in data[smi_col]:
            # Create references for smiles, molecules, fingerprints
            ref_smiles = compound
            ref_mol = Chem.MolFromSmiles(ref_smiles)

            # generate morgan fingerprints
            fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)

            # get reference fingerprint
            ref_fp = fp_generator.GetFingerprint(ref_mol)
            bulk_fp = [fp_generator.GetFingerprint(x) for x in data['ROMol']]

            # Similarity to reference molecule
            similarity = [DataStructs.FingerprintSimilarity(ref_fp, x) for x in bulk_fp]

            # Append list with similarity score
            Tanimoto.append(similarity)

        # Create list of compound name
        compound_name = data[name_col].tolist()

        # Loop Tanimoto Score to each Compound name
        for score in Tanimoto:
            data[compound_name] = Tanimoto

        # generate matrix, only keep columns matching compound name from list
        matrix = data[compound_name]

        # Rename the index using a dictionary comprehension
        new_index_mapping = {old_index: new_name for old_index, new_name in enumerate(compound_name)}
        matrix = matrix.rename(index=new_index_mapping)

        # set instance variable
        self.matrix = matrix

        return matrix


class LibrarySpace:
    """
    Scripts to calculate Rule of 5 or Rule of 3 of chemical library
    """

    def __init__(self, data: pd.DataFrame = None, smi_col: str = None, name_col: str = None):
        self.data = data
        self.smi_col = smi_col
        self.name_col = name_col

    def ro5(self, data: pd.DataFrame = None, smi_col: str = None, generate_descript: bool = False):
        """
        Generate hist plots of chemical space and if compounds violate Rule of 5.
        :param data: pd.DataFrame
            Input pd.DataFrame of molecules. The table must contain at least a column of smiles string.
        :param smi_col: str
            Column containing smiles string from the input pd.DataFrame.
        :param generate_descript: bool
            Whether to generate descriptive tables. This will include 'mw', 'n_lipinski_hda', 'n_lipinski_hdb', and
            'clogp'. Information will then be concat to input pd.DataFrame.
        :return:
        """
        # match instance variable
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smi_col

        # check if columns are in the table
        ro5_cols = ['mw', 'n_lipinski_hda', 'n_lipinski_hdb', 'clogp']
        column_check = [col for col in ro5_cols if col not in data.columns]

        if all(col in data.columns for col in column_check):
            print('everything looks good')

        # generate ro5 descriptions and append results to data
        elif generate_descript is True:
            if smi_col is None:
                raise ValueError('What column is the smiles string?')

            # initialize tqdm for pandas and Score from piki_toolsets
            tqdm.pandas(desc="Calculating Rule of 5 ")
            score = Score()

            # calculate descriptors and keep only relevant to ro5
            descriptors = dm.descriptors.batch_compute_many_descriptors(data[smi_col].apply(dm.to_mol).tolist(),
                                                                        progress=True)
            cols_to_keep = ['mw', 'n_lipinski_hba', 'n_lipinski_hbd', 'clogp', 'qed']
            descriptors = descriptors[cols_to_keep]
            # add descriptors and calculate ro5
            data_df = pd.concat([data, descriptors], axis=1)
            data_df['ro5'] = data_df[smi_col].progress_apply(score.ro5)

            return data_df

        # dynamically print which columns are missing
        else:
            raise ValueError(f"Missing column with header: {', '.join(column_check)}")

    def ro3(self, data: pd.DataFrame = None, smi_col: str = None, generate_descript: bool = False):
        """
        Generate hist plots of chemical space and if compounds violate Rule of 3.
        :param data: pd.DataFrame
            Input pd.DataFrame of molecules. The table must contain at least a column of smiles string.
        :param smi_col: str
            Column containing smiles string from the input pd.DataFrame.
        :param generate_descript: bool
            Whether to generate descriptive tables. This will include 'mw', 'n_lipinski_hda', 'n_lipinski_hdb', and
            'clogp'. Information will then be concat to input pd.DataFrame.
        :return:
        """
        # match instance variable
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smi_col

        # check if columns are in the table
        ro5_cols = ['mw', 'n_lipinski_hda', 'n_lipinski_hdb', 'clogp']
        column_check = [col for col in ro5_cols if col not in data.columns]

        if all(col in data.columns for col in column_check):
            print('everything looks good')

        # generate ro5 descriptions and append results to data
        elif generate_descript is True:
            if smi_col is None:
                raise ValueError('What column is the smiles string?')

            # initialize tqdm for pandas and Score from piki_toolsets
            tqdm.pandas(desc="Calculating Rule of 3 ")
            score = Score()

            # calculate descriptors and keep only relevant to ro5
            descriptors = dm.descriptors.batch_compute_many_descriptors(data[smi_col].apply(dm.to_mol).tolist(),
                                                                        progress=True)
            cols_to_keep = ['mw', 'n_lipinski_hba', 'n_lipinski_hbd', 'clogp', 'qed']
            descriptors = descriptors[cols_to_keep]
            # add descriptors and calculate ro3
            data_df = pd.concat([data, descriptors], axis=1)
            data_df['ro5'] = data_df[smi_col].progress_apply(score.ro3)

            return data_df

        # dynamically print which columns are missing
        else:
            raise ValueError(f"Missing column with header: {', '.join(column_check)}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
