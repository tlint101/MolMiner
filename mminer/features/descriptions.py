"""
Script to encode molecules
"""

import pandas as pd
from tqdm import tqdm

__all__ = ["Frag_Matching"]


class Frag_Matching:
    def __init__(self, data: pd.DataFrame = None, group_col: str = None, activity_col: str = None):
        """
        Initialize class
        :param df:pd.DataFrame
            Input data. Should be in pd.DataFrame format.
        :param group_col: str
            Name of column to process fragments.
        :param activity_col: str
            Name of column to with activity label
        """
        self.data = data
        self.group_col = group_col
        self.activity_col = activity_col

    def kinase_description(self, data: pd.DataFrame = None, group_col="Kinase name", activity_col="Activity", log=False):
        """
        Function to describe kinase dataset from CSV files.
        :param df:pd.DataFrame
            Input data. Should be in pd.DataFrame format.
        :param group_col: str
            Name of column to process fragments.
        :param activity_col: str
            Name of column to with activity label
        :param log: bool
            Optional. Create a log file of kinase descriptions.
        :return:
        """
        # instance variable matching
        if data is None:
            data = self.data
        if group_col is None:
            group_col = self.group_col
        if activity_col is None:
            activity_col = self.activity_col

        # Obtain kinase group
        group_list = data[group_col].unique()
        desc = f"Processing Data By {group_col}"

        for group in tqdm(group_list, desc=desc):
            group_df = data[data[group_col] == group]
            active = group_df[group_df[activity_col] == 1]
            active_per = round((len(active) / (len(group_df)) * 100), 2)
            inactive = group_df[group_df[activity_col] == 0]
            inactive_per = round((len(inactive) / (len(group_df)) * 100), 2)

            if log is True:
                # Ensure we are using the specific logger from the kinome module
                # module is imported here to avoid unwanted generation of log files
                from mminer.logger import kinome
                kinome.log_info(msg=f"{group}")
                kinome.log_info(msg=f"Table Length - {len(group_df)}")
                kinome.log_info(msg=f"Active - {len(active)} ({active_per:.2f}%)")
                kinome.log_info(msg=f"Inactive - {len(inactive)} ({inactive_per:.2f}%) \n")

    def frag_similarity(self, df):
        pass

    def frag_count(self, df):
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
