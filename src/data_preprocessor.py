import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
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
        self.scaler = MinMaxScaler()
        # self._validate_df()

    def __validate_df(self):
        """
        Validate that df is not empty and all columns are in place
        """
        return True
    
    def __map_categorical(self, col: pd.Series, custom_mapping:Optional[dict]=None)->pd.Series:
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
    
    def __binary_encode(self, col:pd.Series)->pd.Series:
        """
        Returns a pd.Df that has to be concatted to the main df using the following:
        output_df = pd.concat([output_df, return_value], axis=1)
        """
        binary_encoder = BinaryEncoder(cols=['churn_category'])
        binary_encoder.fit_transform(col)
        churn_cat = binary_encoder.transform(col)
        return churn_cat
    
    def __scale_numerical(self,col:pd.Series)->pd.Series:
        """
        Scales numerical columns using the same scaler object associated with this class.
        """
        return self.scaler.fit_transform(col)
        
    
if __name__ == "__main__":
    ACC_PATH = "research/data_given/1_account.csv"
    ACC_DF = pd.read_csv(ACC_PATH)
    # print(ACC_DF.head())
    dp = DataPreprocessor(ACC_DF)
 

