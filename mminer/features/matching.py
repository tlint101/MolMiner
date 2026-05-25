"""
The following script will match a query compound with another. The script will primarily be used for fragments and molecules,
but can be used for molecule to molecule matching if needed
"""

import re
import pandas as pd
import safe as sf
from rdkit import Chem
from rdkit.Chem import DataStructs
from mminer.features import Fingerprint
from tqdm import tqdm
from typing import Optional, Union

__all__ = ["Matching"]


class Matching:
    def __init__(self, mol_df: Optional[pd.DataFrame] = None, frag_df: Optional[pd.DataFrame] = None,
                 mol_col: Optional[str] = None, frag_col: Optional[str] = None):
        """
        Initialize the Matching class. Params are optional.
        :param mol_df: Optional[pd.DataFrame]
            DataFrame containing full molecules.
        :param frag_df: Optional[pd.DataFrame]
            DataFrame containing fragments.
        :param mol_col: Optional[str]
            Indicate the column containing smiles string from the mol_df
        :param frag_col: Optional[str]
            Indicate the column containing smiles string from the frag_df
        """
        self.mol_df = mol_df
        self.frag_df = frag_df
        self.mol_col = mol_col
        self.frag_col = frag_col

    @staticmethod
    def sanitize_frag(frag: Optional[Union[str, list]] = None, remove_dummies: bool = False, verbose: bool = False):
        """
        Used to sanitize BRICS fragments so matching to query molecules can be performed. Function will remove the
        associated BRICS numbering and return the * key instead.
        :param frag: Optional[Union[str, list]]
            A list of BRICS fragmented generated from above
        :param remove_dummies: bool
            Use SAFE to remove dummy atoms. This will return smiles string with the * connection removed
        :param verbose: bool
            Output a tqdm progress bar.
        :return: return list of sanitized fragments as smiles strings
        """
        if isinstance(frag, str):
            frag_list = [frag]
        else:
            frag_list = frag

        sanitized_list = []
        if remove_dummies:
            if verbose:
                for frag in tqdm(frag_list, desc="Removing BRICS Numbering: "):
                    sanitized = sf.decode(frag, as_mol=False, remove_dummies=True)
                    sanitized_list.append(sanitized)
            else:
                for frag in frag_list:
                    sanitized = sf.decode(frag, as_mol=False, remove_dummies=True)
                    sanitized_list.append(sanitized)
        else:
            if verbose:
                for frag in tqdm(frag_list, desc="Removing BRICS Numbering: "):
                    # Pattern for fragment sanitization
                    pattern = r"(\[\d+\*?\])"
                    sanitized = re.sub(pattern, r"[*]", frag)
                    sanitized_list.append(sanitized)
            else:
                for frag in frag_list:
                    # Pattern for fragment sanitization
                    pattern = r"(\[\d+\*?\])"
                    sanitized = re.sub(pattern, r"[*]", frag)
                    sanitized_list.append(sanitized)

        return sanitized_list

    def match_counts(
            self, mol_df: Optional[pd.DataFrame] = None, frag_df: Optional[pd.DataFrame] = None,
            mol_col: Optional[str] = None, frag_col: Optional[str] = None, frag_label: str = None
    ):
        """
        Script will count if matches for a fragment found in a query molecule. Molecules and fragments should be given
        as DataFrame
        :param mol_df: Optional[pd.DataFrame]
            DataFrame containing full molecules.
        :param frag_df: Optional[pd.DataFrame]
            DataFrame containing fragments.
        :param mol_col: str
            Column from mol_df containing molecule smiles.
        :param frag_col: str
            Column from frag_df containing fragment smiles.
        :param frag_label: str
            Column containing Fragment Name. This will return fragment name in DataFrame.
        :return: DataFrame containing input mol_df with fragment matches appended.
        """
        # match to instance variable as needed
        if mol_df is None:
            mol_df = self.mol_df
        if frag_df is None:
            frag_df = self.frag_df
        if mol_col is None:
            mol_col = self.mol_col
        if frag_col is None:
            frag_col = self.frag_col

        # input check
        if mol_df is None:
            raise ValueError('Missing mol_df as input!')
        if frag_df is None:
            raise ValueError('Missing frag_df as input!')

        mol_list = mol_df[mol_col].tolist()
        frag_list = frag_df[frag_col].tolist()
        results = []

        for mol in tqdm(mol_list, desc="Matching Fragments to Query Molecule: "):
            frag_match = []
            for frag in frag_list:
                molecule = Chem.MolFromSmiles(mol)
                query = Chem.MolFromSmarts(frag)  # fragments must be read as smarts

                result = molecule.GetSubstructMatches(query)

                frag_match.append(len(result))

            results.append(frag_match)  # append list to outer list

        mol_df["Match Count"] = results

        # Explode the list column into multiple columns
        explode_df = mol_df["Match Count"].apply(pd.Series)

        # Rename the columns in the exploded DataFrame
        explode_df.columns = [f"{frag}" for frag in frag_df[frag_label].tolist()]

        df = pd.concat([mol_df, explode_df], axis=1)

        # drop similarity_array column, comment out for troubleshooting
        df = df.drop(columns=['Match Count'])

        return df

    def similarity(self, mol_df: pd.DataFrame = None, frag_df: pd.DataFrame = None, mol_col: Optional[str] = None,
                   frag_col: Optional[str] = None, frag_label: str = None, chirality: bool = False, **kwargs
                   ):
        """
        Calculate similarity between molecule and query fragments. Inputs must have molecules and fragments in
        pd.DataFrame format.
        :param mol_df: pd.DataFrame
            DataFrame containing full molecules. Must contain a column with ID and a column with smiles strings.
        :param frag_df: pd.DataFrame
            DataFrame containing the fragment query. Must contain a column with ID and a column with smiles strings.
        :param mol_col: str
            The name of the column containing molecule ID for the mol_df.
        :param frag_col: str
            The name of the column containing fragment smiles for the frag_df.
        :param frag_label: str
            The name of the column containing Fragment Name for the frag_df.
        :param chirality: bool
            Include chirality in the fingerprint.
        :param kwargs: Additional optional parameters
            Additional parameters for the fp_generator.smi_to_fp() method.
        :return:
        """
        # match to instance variable as needed
        if mol_df is None:
            mol_df = self.mol_df
        if frag_df is None:
            frag_df = self.frag_df
        if mol_col is None:
            mol_col = self.mol_col
        if frag_col is None:
            frag_col = self.frag_col

        # input check
        if mol_df is None:
            raise ValueError('Missing mol_df as input!')
        if frag_df is None:
            raise ValueError('Missing frag_df as input!')

        mol_list = mol_df[mol_col].tolist()
        frag_list = frag_df[frag_col].tolist()
        sim_results = []

        for mol in tqdm(mol_list, desc="Matching Fragments to Query Molecule: "):
            frag_match = []
            for frag in frag_list:
                # generate fp for molecule and fragment
                fp_generator = Fingerprint()
                mol_fp = fp_generator.smi_to_fp(mol, bitvector=True, chirality=chirality, **kwargs)
                query_fp = fp_generator.smi_to_fp(frag, bitvector=True, chirality=chirality, **kwargs)

                # calculate sim score
                result = DataStructs.TanimotoSimilarity(mol_fp, query_fp)

                frag_match.append(result)

            sim_results.append(frag_match)  # append list to outer list

        mol_df["similarity_array"] = sim_results

        # Explode the list column into multiple columns
        explode_df = mol_df["similarity_array"].apply(pd.Series)

        # Rename the columns in the exploded DataFrame
        explode_df.columns = [f"{frag}" for frag in frag_df[frag_label].tolist()]

        df = pd.concat([mol_df, explode_df], axis=1)

        # drop similarity_array column, comment out for troubleshooting
        df = df.drop(columns=['similarity_array'])

        return df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
