"""
Fragment molecules using BRICS algorithm in RDKit
"""

import re
import pandas as pd
from mminer.features import Score
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm
import datamol as dm
import safe as sf
import warnings
# 1. Move this to the absolute top, before any other imports
warnings.filterwarnings("ignore", message=".*is part of SAFedoubeHeadsModel.forward's signature.*")

__all__ = ["Fragmentation"]


class Fragmentation:
    def __init__(self, data: pd.DataFrame = None, name_col: str = None, smi_col: str = None, target_col: str = None,
                 method: str = 'brics', safe_encoding: bool = True):
        """
        parameters for Fragmentation initialization. Params here are optional.
        :param data: pd.DataFrame
            Input data in pd.DataFrame format.
        :param name_col: str
            Column header to designate names of molecules.
        :param smi_col: str
            Column header to for data containing smiles string.
        :param target_col: str
            Column header to designate target molecules.
        :param method: str
            Indicate the splitting method. Only
        :param safe_encoding: bool
            Set if safe_encoding is used or if RDKit fragmenting is used instead.
        """
        self.data = data
        self.name_col = name_col
        self.smiles_col = smi_col
        self.target_col = target_col
        self.method = method
        self.safe_encoding = safe_encoding

    # todo Add Multiprocessing!!!
    # todo Add Traceback!!!
    # todo references for linker/decorator generation
    # https://iwatobipen.wordpress.com/2020/01/23/cut-molecule-to-ring-and-linker-with-rdkit-rdkit-chemoinformatics-memo/
    # https://pubs.acs.org/doi/10.1021/acs.jcim.6b00596
    # Using joblib in the list? double check on usage
    def brics_fragment(
            self,
            data: pd.DataFrame = None,
            name_col: str = None,
            smiles_col: str = None,
            target_col: str = None,
            verbose: bool = False,
            n_jobs: int = -1
    ):
        """
        Fragment molecules using RDKit directly. Function will take in a dataframe. Must contain a column with smiles
        string and a column with compound ID. Smiles string will be translated into a RDKit mol object and then
        fragmented using the BRICS method. The fragments will be mapped back to the input dataframe.

        :param data: pd.DataFrame
            Input data. Should be in pd.DataFrame format. 
        :param name_col: str
            Name Column.
        :param smiles_col: str
            Smiles column.
        :param target_col: str
            Target column (Optional)
        :param verbose: bool
            Print Compound ID and fragment number during fragmentation.
        :param n_jobs: int
            Set the number of CPUs to use. -1 uses all.
        :return:
        """
        # check instance variables
        if data is None:
            data = self.data
        if name_col is None:
            name_col = self.name_col
        if smiles_col is None:
            smiles_col = self.smiles_col
        if target_col is None:
            target_col = self.target_col

        # Convert smiles_col and name_col into a list
        smi_list = data[smiles_col].to_list()
        name_list = data[name_col].to_list()

        # Optional list for compound target
        if target_col is not None:
            target_list = data[target_col].to_list()
        else:
            target_list = None

        # Loop through smi_list. Encode, fragment, and decode fragments
        fragment_list = []
        mol_smiles_list = []
        compound_name_list = []
        final_target_list = []

        # # Section to parallelize function
        # # Parallelize the processing of DataFrame using joblib with specified number of CPUs
        # fragment_list = Parallel(n_jobs=n_jobs)(delayed(self.brics_fragmentation_logic)(row[smiles_col]) for _, row in df.iterrows())

        # frag_df = pd.DataFrame(
        #     {'origin': compound_name_list, 'target': final_target_list, 'ori_smiles': mol_smiles_list,
        #      'frag_smi': fragment_list})

        if target_list is not None:
            _brics_fragmentation_logic(
                smi_list,
                name_list,
                target_list,
                fragment_list,
                mol_smiles_list,
                compound_name_list,
                final_target_list,
                verbose,
                n_jobs,
            )
            frag_df = pd.DataFrame(
                {
                    "origin": compound_name_list,
                    "target": final_target_list,
                    "ori_smiles": mol_smiles_list,
                    "frag_smi": fragment_list,
                }
            )
        else:
            _brics_fragmentation_logic(
                smi_list,
                name_list,
                target_list,
                fragment_list,
                mol_smiles_list,
                compound_name_list,
                final_target_list,
                verbose,
                n_jobs,
            )
            frag_df = pd.DataFrame(
                {
                    "origin": compound_name_list,
                    "ori_smiles": mol_smiles_list,
                    "frag_smi": fragment_list,
                }
            )

        return frag_df

    # todo add scripts for recap fragmentation and building from recap fragment
    # Check https://iwatobipen.wordpress.com/2020/10/16/easy-way-to-connect-fragments-rdkit-tips-memo/
    def recap_fragment(self):
        raise Exception("RDKit recap fragmentation not implemented yet")

    def _recap_fragment_logic(self):
        raise Exception("RDKit recap fragmentation not implemented yet")

    """
    Functions below use SAFE to fragment. I am leaving it as I may end up using it in the future after troubleshooting
    different fragmentation methods. 
    """

    def safe_fragmentation(
            self,
            data: pd.DataFrame = None,
            name_col: str = None,
            smiles_col: str = None,
            method: str = None,
            target_col: str = None,
            dummies: bool = False,
            ignore_stereo: bool = False,
            desc: str = None,
            verbose: bool = False,
            log: bool = False
    ):
        """
        Take in a dataframe. Must contain a column with smiles string and a column with compound ID. Smiles string will
        be translated into a RDKit mol object and then fragmented using the given method. The fragments will be mapped
        back to the input dataframe.

        :param data: pd.DataFrame
            pd.DataFrame containing a smiles string and compound ID.
        :param name_col: str
            The name_col is used to map where the compound where the fragment originated.
        :param target_col: str
            The target_col can the compound name or target name. The name will be mapped back to the fragments.
        :param smiles_col: str
            A compound in smiles string format
        :param method: str
            Method for fragmentation. The current available methods are: 'brics', 'recap', 'mmpa', 'hr', 'attach'.
        :param dummies: bool
            Determine whether to generate a column of fragments without dummies.
        :param ignore_stereo: bool
            Option to ignore stereochemistry.
        :param desc: str
            Optional custom description. Default None will write "Fragmenting Compound".
        :param verbose: str
            Print out additional information while running.
        :param log: bool
            Set an output to log molecule that cannot be fragmented.
        :return: pd.DataFrame containing a column with the fragments.
        """
        # check instance variables
        global slicer
        if data is None:
            data = self.data
        if name_col is None:
            name_col = self.name_col
        if smiles_col is None:
            smiles_col = self.smiles_col
        if target_col is None:
            target_col = self.target_col

        if method is None:
            method = self.method

        # set slicing conditions
        if method is not None:
            slicer = (method if method in ("brics", "recap", "mmpa", "hr", "attach") else "brics")

        # Convert smiles_col and name_col into a list
        smi_list = data[smiles_col].to_list()
        compound_list = data[name_col].to_list()

        # Optional list for compound target
        if target_col is not None:
            target_list = data[target_col].to_list()
        else:
            target_list = None

        # Loop through smi_list. Encode, fragment, and decode fragments
        fragment_list = []
        mol_smiles_list = []
        compound_name_list = []
        final_target_list = []
        encoded_frag_list = []

        if target_list is not None:
            _safe_fragmentlogic(
                compound_list,
                compound_name_list,
                final_target_list,
                fragment_list,
                mol_smiles_list,
                slicer,
                smi_list,
                target_list,
                encoded_frag_list,
                ignore_stereo,
                desc,
                verbose,
                log
            )
            # # Debug information
            # print(f"Length of compound_name_list: {len(compound_name_list)}")
            # print(f"Length of final_target_list: {len(final_target_list)}")
            # print(f"Length of mol_smiles_list: {len(mol_smiles_list)}")
            # print(f"Length of fragment_list: {len(fragment_list)}")
            # print(f"Length of encoded_frag_list: {len(encoded_frag_list)}")

            # Make sure all lists have the same length
            min_length = min(len(compound_name_list), len(final_target_list),
                             len(mol_smiles_list), len(fragment_list), len(encoded_frag_list))

            frag_df = pd.DataFrame(
                {
                    "origin": compound_name_list[:min_length],
                    "target": final_target_list[:min_length],
                    "smiles": mol_smiles_list[:min_length],
                    "frag_smi": fragment_list[:min_length],
                    "encoded_frag": encoded_frag_list[:min_length],
                }
            )
        else:
            _safe_fragmentlogic(
                compound_list,
                compound_name_list,
                final_target_list,
                fragment_list,
                mol_smiles_list,
                slicer,
                smi_list,
                target_list,
                encoded_frag_list,
                ignore_stereo,
                desc,
                verbose,
                log
            )

            # Make sure all lists have the same length
            min_length = min(len(compound_name_list), len(mol_smiles_list),
                             len(fragment_list), len(encoded_frag_list))

            frag_df = pd.DataFrame(
                {
                    "origin": compound_name_list[:min_length],
                    "smiles": mol_smiles_list[:min_length],
                    "frag_smi": fragment_list[:min_length],
                    "encoded_frag": encoded_frag_list[:min_length],
                }
            )

        # remove_dummies
        if dummies is True:
            frag_smi_list = frag_df['frag_smi'].to_list()
            no_dummies_list = self.decode_frag(frag_smi_list, remove_dummies=True)

            frag_df['no_dummies'] = no_dummies_list

        return frag_df

    def decode_frag(self, safe_list: list = None, as_mol: bool = False, remove_dummies: bool = False):
        """
        Decode a list of SAFE strings. This will return a list of decoded strings or RDKit Molecules
        :param safe_list: list
            List of fragments as SAFE strings
        :param as_mol: bool
            Convert fragment smi into RDKit Molecules
        :param remove_dummies: bool
            Keep or remove attachment points

        :return: A list of molecules or RDKit Molecules
        """

        frag_list = []

        for frag in tqdm(safe_list, desc="Decoding fragments: "):
            decode = sf.decode(frag, as_mol=as_mol, remove_dummies=remove_dummies)
            frag_list.append(decode)

        return frag_list

    # todo generate code to slice molecules as head-linker-tail format
    # https://safe-docs.datamol.io/stable/api/safe.html#safe.utils.MolSlicer
    def slicer(self):
        pass

    # todo add documentation and condense code where possible

    def scale(self, data: pd.DataFrame = None, smi_col: str = None, name_col: str = None,
              target_col: str = None, activity_col: str = None, ring_mining: bool = True,
              active_cutoff: float = 0.1, inactive_cutoff: float = 0.8, rules: str = 'ro3',
              ignore_stereo: bool = False, log: bool = False):

        """
        To scale the fragment results. Output will be a pd.DataFrame containing the fragment smiles, its molecule of
        origin, and how often it appears in active/inactive datasets.
        :param data: pd.DataFrame
            Input data in pd.DataFrame format.
        :param smi_col: str
            Input data column containing smile string.
        :param name_col: str
            Input data column containing molecule name or ID.
        :param target_col: str
            Input data column containing target column i.e. protein or molecule class.
        :param activity_col: str
            Input data column containing the activity column.
        :param ring_mining: bool
            Determine to output only fragments with rings.
        :param active_cutoff: float
            Cutoff for how often a fragment must appear in the active set.
        :param inactive_cutoff: float
            Cutoff for how often a fragment must appear in the inactive set.
        :param rules: str
            Set the fragment rules. Only Rule of 3 ('ro3') or Rule of 5 ('ro5') available.
        :param ignore_stereo: bool
            Option to ignore stereochemistry.
        :param log: bool
            Set an output to log molecule that cannot be fragmented.
        :return:
        """

        # match instance variables
        safe_encoding = self.safe_encoding
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smiles_col

        # split data input by activity
        active_df = data[data[activity_col] == 1]
        inactive_df = data[data[activity_col] == 0]

        # desc messages
        active_desc = "Fragmenting Active Compounds"
        inactive_desc = "Fragmenting Inactive Compounds"

        # fragment molecules
        if self.safe_encoding is True:
            fragmentation = Fragmentation(smi_col=smi_col, name_col=name_col, target_col=target_col)
            active_df = fragmentation.safe_fragmentation(active_df, dummies=True, desc=active_desc,
                                                         ignore_stereo=ignore_stereo, log=log)
            inactive_df = fragmentation.safe_fragmentation(inactive_df, dummies=True, desc=inactive_desc,
                                                           ignore_stereo=ignore_stereo, log=log)
        else:
            raise ValueError(
                'Only safe_encoding is available for FragmentFilter.scale()!')

        # count fragments by active and inactive dataframes first.
        scale = FragmentCount(safe_encoding=safe_encoding)
        active_df = scale.count_frag(active_df, frag_col="no_dummies")
        inactive_df = scale.count_frag(inactive_df, frag_col="no_dummies")

        # set activity column
        active_df[activity_col] = 1
        inactive_df[activity_col] = 0

        # combine and count fragments
        frag_df = pd.concat([active_df, inactive_df]).reset_index(drop=True)

        # count fragments again after combining active and inactive dataframes
        frag_df = scale.count_frag(frag_df, frag_col="no_dummies", output_name='total_count')

        # remove duplicate fragments
        frag_df = scale.remove_frag_dup(frag_df, smi_col="no_dummies", name_col='origin',
                                        activity_col=activity_col)

        # scale and only keep active
        frag_df = scale.scale_frag(frag_df, smi_col='no_dummies', active_cutoff=active_cutoff,
                                   activity_col=activity_col, inactive_cutoff=inactive_cutoff,
                                   ring_mining=ring_mining, rules=rules)

        return frag_df


"""
FragmentCount used to count smiles string in a pd.DataFrame. Methods here has been converted into the "scale()" method 
in class FragmentFilter.
"""


class FragmentCount:
    def __init__(self, data: pd.DataFrame = None, smiles_col: str = None, safe_encoding: bool = False):
        self.data = data
        self.smiles_col = smiles_col
        self.safe_encoding = safe_encoding

    def descriptors(self, data: pd.DataFrame = None, smi_col: str = None, **kwargs):
        """
        Rule for filtering the fragments. This will follow the Rule of 3 criteria.
        Input is a DataFrame containing a smiles string for each fragment. This will be converted into a Datamol/RDKit
        molecule

        :param data: pd.DataFrame
            DataFrame containing fragments
        :param smi_col: str
            Column with smiles string
        :param kwargs: keyword arguments to filter rows based off of a value in a given column. For Rule of 3, the
        following is used:
            n_rings=('>=', 1), n_rotatable_bonds=('<=', 3), clogp=('<=', 3), mw=('<=', 300), n_lipinski_hba=('<=', 3),
            n_lipinski_hbd=('<=', 3)
        :return: DataFrame
        """
        # match instance variable
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smiles_col

        working_df = data.copy()
        working_df["frag"] = working_df[smi_col].apply(dm.to_mol)
        frag_list = working_df["frag"].to_list()

        # Calculate descriptors
        if "progress" in kwargs:
            progress = kwargs["progress"]
            descriptors_df = dm.descriptors.batch_compute_many_descriptors(
                frag_list, progress=progress
            )
        else:
            descriptors_df = dm.descriptors.batch_compute_many_descriptors(
                frag_list, progress=True
            )

        # Drop frag ROMol col
        working_df.drop("frag", axis=1, inplace=True)

        # Combine df
        frag_df = pd.concat([working_df, descriptors_df], axis=1)

        return frag_df  # Return original DataFrame if no filters specified

    def count_frag(self, data: pd.DataFrame = None, frag_col=None, output_name=None):
        """
        Function to count strings. This is primarily used to count fragment smiles string, but it can also be used to
        count any column containing strings
        :param data: pd.DataFrame
            Input data. Should be in pd.DataFrame format.
        :param frag_col: str
            Column header containing string for counting
        :param output_name: str
            Output Name is optional. This will give a specific column header for count column. The counts are based on
            smiles string appearances as indicated from the frag_col. If no name is given, it will default to
            "appearance".

        :return: DataFrame with additional columns with count results
        """
        # match instance variable
        if data is None:
            data = self.data

        if output_name is None:
            output_name = "appearance"
        if frag_col is None:
            raise ValueError("Designate column name with smiles strings")
        data[output_name] = data.groupby(frag_col)[frag_col].transform("count")
        return data

    def remove_frag_dup(
            self,
            data: pd.DataFrame = None,
            smi_col: str = None,
            name_col: str = 'origin',
            activity_col: str = "activity",
    ):
        """
        Remove duplicate fragment smiles strings from DataFrame
        :param data: pd.DataFrame
            Input data. Should be in pd.DataFrame format.
        :param smi_col: str
            String with smiles column name
        :param name_col: str
            String with name of origin compound.
        :param activity_col: str
            String containing header for activity column
        :return: DataFrame
        """
        # match instance variable
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smiles_col

        # remove duplicates
        duplicate_df = data.copy()
        duplicate_df = duplicate_df.drop_duplicates(subset=[smi_col, activity_col]).reset_index(drop=True)
        # drop origin column
        duplicate_df = duplicate_df.drop(columns=['origin'])

        # Create a dataframe containing only frag_smi and list of compound origin name. Origin column will be renamed.
        origin_list = data.groupby(smi_col)[name_col].apply(list).reset_index()
        # remove duplicate in a list for each row in column
        origin_list[name_col] = origin_list[name_col].apply(lambda x: list(set(x)))

        # Merge no_duplicates and origin_list on frag_smi
        output = pd.merge(duplicate_df, origin_list, on=smi_col, how="left")

        # Drop a duplicate origin list that arises due to pd.merge()
        output = output[~output.duplicated(subset='origin', keep='first')]

        return output

    def scale_frag(
            self,
            data: pd.DataFrame = None,
            smi_col: str = None,
            active_cutoff: float = 0.2,
            inactive_cutoff: float = 0.5,
            ring_mining: bool = True,
            rules: str = None,
            activity_col: str = "activity",
            percentage_col: str = "percentage",
            frag_count: str = "appearance",
            total_count: str = "total_count",
            target: str = "target"
    ):
        """
        Function to scale the fragments. This function needs to have a df with specific columns, preferably one
        generated using the remove_frag_dup() function. This function will generate a percentage of the fragment
        appearance in the active and inactive set. These results can then be scaled, meaning the fragments are kept if
        they appear more than X times in the active set and no more than X times in the inactive set.

        :param data: pd.DataFrame
            Input data. Should be in pd.DataFrame format.
        :param smi_col: str
            Column with fragment smiles string.
        :param active_cutoff: float
            Set percentage of fragment appearance in the active set.
        :param inactive_cutoff: float
            Set percentage of fragment appearance in the inactive set.
        :param ring_mining: bool
            If true, only fragments with rings will be kept.
        :param rules: str
            Set the rule filtering. Only Rule of 3 (ro3) or Rule of 5 (ro5) are available.
        :param activity_col: str
            Column with activity label.
        :param percentage_col: str
            Column with appearance percentage.
        :param frag_count: str
            Column with fragment appearance count based on activity.
        :param total_count: str
            Column with total fragment appearance count.
        :param target: str
            Column containing fragment target. This can be adjusted for kinase group or specific protein target from the
            input DataFrame.
        :return:
        """
        # match instance variable
        if data is None:
            data = self.data
        if smi_col is None:
            smi_col = self.smiles_col

        # take only active frags
        data = data[data[activity_col] == 1]

        # using data.copy() to avoid SettingWithCopyWarning
        data = data.copy()

        # Generate percentage of appearance for each fragment
        data[percentage_col] = data[frag_count] / data[total_count]

        # Add a new column 'percentage_active' and 'percentage_inactive' based on the 'activity' column
        data['percentage_active'] = 1 - data[percentage_col]
        data['percentage_inactive'] = 1 - data[percentage_col]

        if target is None:
            target = "target"

        if self.safe_encoding is False:
            filtering_df = data.pivot_table(
                index=smi_col,
                values=[
                    "frag_smi",
                    "target",
                    "origin",
                    "percentage_active",
                    "percentage_inactive",
                    frag_count,
                    "total_count",
                ],
                # aggfunc tells if column should be summed or kept as is
                aggfunc={
                    "frag_smi": "sum",
                    "target": "sum",
                    "origin": "sum",
                    "percentage_active": "max",
                    "percentage_inactive": "max",
                    frag_count: "max",
                    "total_count": "max",
                },
                fill_value=0,
            ).reset_index()
        else:
            filtering_df = data.pivot_table(
                index=smi_col,
                values=[
                    "frag_smi",
                    "encoded_frag",
                    "target",
                    "origin",
                    "percentage_active",
                    "percentage_inactive",
                    frag_count,
                    "total_count",
                ],
                # aggfunc tells if column should be summed or kept as is
                aggfunc={
                    "frag_smi": "sum",
                    "encoded_frag": "sum",
                    "target": "sum",
                    "origin": "sum",
                    "percentage_active": "max",
                    "percentage_inactive": "max",
                    frag_count: "max",
                    "total_count": "max",
                },
                fill_value=0,
            ).reset_index()

        # fix percentage for consistency (issue may be from the pivot. Fix but later)
        filtering_df['percentage_active'] = filtering_df[frag_count] / filtering_df['total_count']
        filtering_df['percentage_inactive'] = 1 - filtering_df['percentage_active']

        # filter criteria
        filtered_df = filtering_df[filtering_df["percentage_active"] >= active_cutoff].reset_index(drop=True)
        filtered_df = filtered_df[filtered_df["percentage_inactive"] <= inactive_cutoff].reset_index(drop=True)

        # Fix target column to include only 1 string for target and reorder columns to my preference
        unique_names = data[target].unique()
        filtered_df[target] = unique_names[0]
        if self.safe_encoding is True:
            columns_order = [
                smi_col,
                "frag_smi",
                "encoded_frag",
                "target",
                "origin",
                "percentage_active",
                "percentage_inactive",
                frag_count,
                "total_count",
            ]
            filtered_df = filtered_df[columns_order]
        else:
            columns_order = [
                smi_col,
                "frag_smi",
                "target",
                "origin",
                "percentage_active",
                "percentage_inactive",
                frag_count,
                "total_count",
            ]
            filtered_df = filtered_df[columns_order]

        # Calculate descriptors for fragments
        desc_df = self.descriptors(filtered_df, smi_col=smi_col, progress=False)

        # Filter by rules
        if rules == "ro3":
            tqdm.pandas(desc="Calculating Rule of 3")

            scores = Score()

            cols_to_keep = ['mw', 'n_lipinski_hba', 'n_lipinski_hbd', 'clogp', 'qed', 'n_rings']
            descriptors = desc_df[cols_to_keep]
            # add descriptors and calculate ro5
            data_df = pd.concat([filtered_df, descriptors], axis=1)
            data_df['ro3'] = data_df['frag_smi'].progress_apply(scores.ro3)
            rules_df = data_df

        elif rules == "ro5":
            tqdm.pandas(desc="Calculating Rule of 5")

            scores = Score()

            cols_to_keep = ['mw', 'n_lipinski_hba', 'n_lipinski_hbd', 'clogp', 'qed', 'n_rings']
            descriptors = desc_df[cols_to_keep]
            # add descriptors and calculate ro5
            data_df = pd.concat([filtered_df, descriptors], axis=1)
            data_df['ro5'] = data_df['frag_smi'].progress_apply(scores.ro5)
            rules_df = data_df

        else:
            rules_df = filtered_df

        if ring_mining is True:
            rules_df = rules_df[rules_df["n_rings"] >= 1].reset_index(drop=True)

        return rules_df


"""
Extract fragments by group header from csv table
"""


class FragmentExtraction:
    def __init__(self, data: pd.DataFrame = None, target_col: str = None, group_list: list = None):
        """
        Initialize FragmentExtraction class

        :param data: pd.DataFrame
            Dataset containing fragment information. Should be a .csv file generated from the fragmentation.py.
        :param target_col: str
            Name of fragment group to extract.
        :param group_list: str
            A list of all fragment groups from the .csv file.
        """
        self.data = data
        self.target = target_col
        self.group_list = group_list

    # todo official target_cutoff number
    def extract_frag(self, data: pd.DataFrame = None, target_col: str = None, group_list: list = None,
                     target_cutoff: float = 0.5, group_cutoff: float = 0.5):
        """

        :param data: pd.DataFrame
            Dataset containing fragment information. Should be a .csv file generated from the fragmentation.py.
        :param target_col: str
            Name of fragment group to extract.
        :param group_list: str
            A list of all fragment groups from the .csv file.
        :param target_cutoff: float
            Set the appearance cutoff for the target group.
        :param group_cutoff: float
            Set the appearance cutoff for a fragment appearance in additional groups.
        :return:
        """

        # check for instance variables
        if data is None:
            data = self.data
        if target_col is None:
            target_col = self.target
        if group_list is None:
            group_list = self.group_list

        # in order for function to work, it needs to know a list of all items/groups we are sorting.
        if group_list is None and self.group_list is None:
            raise ValueError("Need to give a list of all items of groups for sorting!")

        data = data.copy()  # to keep original data unmodified

        # extract group from list
        query_list = _group_query(target_col, group_list)

        # filter fragment appearance based on how it appears in target group compared to additional groups
        filtered = data[
            (data[target_col] >= target_cutoff) & (data[query_list].max(axis=1) <= group_cutoff)].reset_index(
            drop=True)

        return filtered


# support function to query a fragment group for extract_frag
def _group_query(query: str = None, groups: list = None):
    group_list = groups.copy()  # to keep original list unmodified
    group_list.remove(query)
    return group_list


"""
Scale and generate heatmap of fragments
"""


class VisualizeGlobalFragment:
    def __init__(self, data: pd.DataFrame = None, smiles_col: str = None, safe_encoding: bool = False):
        self.data = data
        self.smiles_col = smiles_col
        self.safe_encoding = safe_encoding


def _remove_attachment_logic(string):
    """logic funciton for removing the number attachmenet. Created to be utilized with the pd.DataFrame.apply."""
    return re.sub(r':\d+', '', string)


def _brics_fragmentation_logic(
        smi_list,
        name_list,
        target_list,
        fragment_list,
        mol_smiles_list,
        compound_name_list,
        final_target_list,
        verbose,
        n_jobs,
):
    """
    Logic for brics_fragment. Includes fragmenting molecules using BRICS.
    """
    # For progress bar description
    desc = "Fragmenting Compound"

    # todo -> the if/else can be placed as its own function and then use joblib to parallel
    # Fragment molecule
    if target_list is not None:
        # # To use joblib, you need to first create a function for the logic first. extract teh try/for logic below first
        # results = Parallel(n_jobs=n_jobs)(delayed(process_item)(smi, name, target) for smi, name, target in tqdm(
        #         zip(smi_list, name_list, target_list), total=len(smi_list), desc=desc
        #     )
        # )
        for smi, name, target in tqdm(
                zip(smi_list, name_list, target_list), total=len(smi_list), desc=desc
        ):
            try:  # to catch exceptions
                mol = Chem.MolFromSmiles(smi)
                fragments = BRICS.BRICSDecompose(
                    mol, returnMols=False
                )  # Get fragments as molecule objects
                fragment_list.extend(fragments)  # Add fragments to the library

                # append compound name to list based on number of generated fragments
                for i in range(len(fragments)):
                    compound_name_list.append(name)
                    mol_smiles_list.append(smi)
                    final_target_list.append(target)

                # Output splitting info
                if verbose is True:
                    print(f"{name} has {len(fragments)} fragments")
            except Exception as e:
                print(f"Cannot Fragment Compound {name} using BRICS, {smi}")
                continue
    else:
        for smi, name in tqdm(
                zip(smi_list, name_list), total=len(smi_list), desc=desc
        ):
            try:  # to catch exceptions
                mol = Chem.MolFromSmiles(smi)
                fragments = BRICS.BRICSDecompose(
                    mol, returnMols=False
                )  # Get fragments as molecule objects
                fragment_list.extend(fragments)  # Add fragments to the library

                # append compound name to list based on number of generated fragments
                for i in range(len(fragments)):
                    compound_name_list.append(name)
                    mol_smiles_list.append(smi)

                # Output splitting info
                if verbose is True:
                    print(f"{name} has {len(fragments)} fragments")
            except Exception as e:
                print(f"Cannot Fragment Compound {name} using BRICS")
                continue


def _safe_fragmentlogic(
        compound_list,
        compound_name_list,
        final_target_list,
        fragment_list,
        mol_smiles_list,
        slicer,
        smi_list,
        target_list,
        encoded_frag_list,
        ignore_stereo,
        desc,
        verbose,
        log
):
    """
    Logic for fragment_mol()
    """
    if desc is None:
        desc = "Fragmenting Compound: "

    if target_list is not None:
        for smi, name, target in tqdm(
                zip(smi_list, compound_list, target_list),
                total=len(smi_list),
                desc=desc,
        ):
            try:  # To catch exceptions
                # encode, split, and decode. Place fragments into a list
                safe_str = sf.encode(smi, slicer=slicer, require_hs=True, ignore_stereo=ignore_stereo)
                safe_fragment = safe_str.split(".")

                for frag in safe_fragment:
                    # Append original compound
                    mol_smiles_list.append(smi)
                    # Append encoded fragment string
                    encoded_frag_list.append(frag)
                    # Decode fragment and append into list
                    frag_string = sf.decode(frag, as_mol=True, remove_dummies=False)
                    fragment_list.append(Chem.MolToSmiles(frag_string))
                    # Add drug/target and original compound smiles string list
                    compound_name_list.append(name)
                    final_target_list.append(target)

                # To output splitting information.
                if verbose is True:
                    print(
                        f"Representation of {name} using {len(safe_str.split('.'))} fragments"
                    )
            except Exception as e:
                print(f"Cannot Fragment Compound {name}: {smi}")
                if log is True:
                    from mminer.logger import frag_error
                    frag_error.log_info(title='fragmentation', msg=f"Cannot Fragment {name}: {smi}")
                continue
    else:
        for smi, name in zip(smi_list, compound_list):
            try:
                # encode, split, and decode. Place fragments into a list
                safe_str = sf.encode(smi, slicer=slicer, ignore_stereo=ignore_stereo)
                safe_fragment = safe_str.split(".")

                for frag in safe_fragment:
                    # Append original compound
                    mol_smiles_list.append(smi)
                    # Append encoded fragment string
                    encoded_frag_list.append(frag)
                    # Decode fragment and append into list
                    frag_string = sf.decode(frag, as_mol=True, remove_dummies=False)
                    fragment_list.append(Chem.MolToSmiles(frag_string))
                    # Add drug/target and original compound smiles string list
                    compound_name_list.append(name)

                # To output splitting information.
                if verbose is True:
                    print(
                        f"Representation of {name} using {len(safe_str.split('.'))} fragments"
                    )
            except Exception as e:
                print(f"Cannot Fragment Compound {name}: {smi}")
                if log is True:
                    from mminer.logger import frag_error
                    frag_error.log_info(title='fragmentation', msg=f"Cannot Fragment {name}: {smi}")
                continue


if __name__ == "__main__":
    import doctest

    doctest.testmod()
