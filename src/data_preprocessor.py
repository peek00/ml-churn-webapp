import pandas as pd
from typing import Optional
class DataPreprocessor:
    """
    Perform data cleaning steps on dataset.
    """

    def __init__(self, df:pd.DataFrame):
        """
        Assume that the table has been fully joined.
        """
        self.df = df
        # self._validate_df()

    def __validate_df(self):
        """
        Validate that df is not empty and all columns are in place
        """
        return True
    
    def __map_categorical(self, col: pd.Series, custom_mapping:Optional[dict]=None):
        """
        Map common values to numerical values. Provide custom mapping if required.
        """
        mapping = {
            "Yes": 1,
            "No": 0,
            "Male": 1,
            "Female": 0
        }
        if custom_mapping:
            return col.map(custom_mapping)
        return col.map(mapping)
        
    
if __name__ == "__main__":
    ACC_PATH = "research/data_given/1_account.csv"
    ACC_DF = pd.read_csv(ACC_PATH)
    # print(ACC_DF.head())
    dp = DataPreprocessor(ACC_DF)
 

