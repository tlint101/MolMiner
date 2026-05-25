import pandas as pd
from tqdm import tqdm
import rdkit.DataStructs
import datamol as dm
from mminer.utils.fingerprint_misc import smiles_to_fp

__all__ = ["Fingerprint"]


class Fingerprint:

    def __init__(self, data: pd.DataFrame = None, smi_col: str = None):
        """
        Initialize the Fingerprint class
        :param data: pd.DataFrame
            DataTable containing information needed for plotting
        :param smi_col: str
             The data from the table corresponding to the molecule smiles string
        """
        self.data = data
        self.smi_col = smi_col

    def smi_to_fp(self,
                  smi: str = None,
                  method: str = "morgan",
                  radius: int = 2,
                  nbits: int = 2048,
                  bitvector: bool = False,
                  chirality: bool = False,
                  ):
        """
        Generate fingerprint from a smiles string. Different fingerprint methods can
        be used depending on the input.
        :param smi: str
            smiles string of the molecule
        :param method: str
            method to use for fingerprint generation.
        :param radius: int
            Set the radius of the circular fingerprint. Default is 2. If a different method is used, this will be
            ignored.
        :param nbits: int
            Number of bits to use for fingerprint. If using MacCSkeys, this will be ignored.
        :param bitvector: bool
            Output fingerprint results as ExplicitBitVect or as np.Array
        :param chirality: bool
            Include chirality in the fingerprint.
        :return: return fingerprint as ExplicitBitVect or np.Array Default returns ExplicitBitVect.
        """

        output = smiles_to_fp(smi=smi, method=method, radius=radius, nbits=nbits, bitvector=bitvector,
                              include_chirality=chirality)

        return output

    def bulk_smi_to_fp(self,
                       data: pd.DataFrame = None,
                       smi_col: str = None,
                       method: str = "morgan",
                       expand_fp: bool = True,
                       radius: int = 2,
                       nbits: int = 2048,
                       bitvector: bool = False,
                       chirality: bool = False):
        """
        Bulk convert smiles string into fingerprints. Can only be used if data is in pd.DataFrame format.
        :param data: pd.DataFrame
            Input data. Should be in pd.DataFrame format.
        :param smi_col: str
            Column designating smiles string.
        :param method: str
            Fingerprint method. Currently only 'Morgan', 'feature_morgan', 'atompair', 'rdkit', 'torsion', and 'maccs'
            are available.
        :param expand_fp: bool
            Set to expand the fingerprint bits into individual columns. Default to True.
        :param radius: int
            Set the radius of the circular fingerprint. Default is 2. If a different method is used, this will be
            ignored.
        :param nbits: int
            Number of bits to use for fingerprint. If using MacCSkeys, this will be ignored.
        :param bitvector: bool
            Output fingerprint results as ExplicitBitVect or as np.Array
        :param chirality: bool
            Include chirality in the fingerprint.
        :return:
        """
        # set to instance variables
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smi_col

        # Generate smiles list and empty list for fp
        smi_list = data[smi_col].tolist()
        fp_list = []

        # Generate fingerprint
        for str in tqdm(smi_list, desc="Generating Fingerprint"):
            fp = self.smi_to_fp(smi=str, method=method, radius=radius, nbits=nbits, bitvector=bitvector,
                                chirality=chirality)
            fp_list.append(fp)
        data['fp'] = fp_list

        # expand fp
        if expand_fp is True:
            data = self.expand_fp(data=data, fp_col='fp')

        return data

    def expand_fp(self, data: pd.DataFrame = None, fp_col: str = None):
        """
        Input a DataFrame with a column containing a fingerprint in a numpy array. Conditional check will ensure that
        fingerprint column is in np.Array format before expanding fingerprint into individual columns.
        :param data: pd.Dataframe
            input dataframe
        :param fp_col: str
            column containing molecular fingerprint
        :return:
        """

        # Conditional check if the fingerprint is an array or ExplicitBitVect
        instance_check = isinstance(
            data[fp_col][0], rdkit.DataStructs.cDataStructs.ExplicitBitVect
        )

        # print("check is:", instance_check)  # for debugging

        if instance_check is True:
            bit_vect = []
            for fp in tqdm(
                    data[fp_col], desc="Converting rdkit ExplicitBitVect to np.Array"
            ):
                array = dm.fp_to_array(fp)
                bit_vect.append(array)
            data[fp_col] = bit_vect
        else:
            # fp_col is already an array!
            fp_col = fp_col

        # Append fingerprint to table
        # Initialize an empty DataFrame to store results
        fp_df_list = []

        # Process each row and convert to DataFrame with progress bar
        for i in tqdm(range(len(data)), desc="Adding Fingerprints to Table"):
            row = data.iloc[i]
            fp_df_list.append(pd.DataFrame([row['fp']], columns=[f'fp_{j}' for j in range(len(row['fp']))]))

        # Concatenate the list of DataFrames
        fp_df = pd.concat(fp_df_list, ignore_index=True)

        # Drop the fp column from the original DataFrame
        data = data.drop(columns='fp')

        # Concatenate the DataFrames with progress bar
        merge_df = pd.concat([data.reset_index(drop=True), fp_df], axis=1)

        return merge_df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
