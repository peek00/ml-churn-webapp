import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from typing import Optional
from pathlib import Path
import pickle
import os

"""
This backend preprocessor was written with the version that uses Catboost and PCA.
"""
class DataPreprocessor:
    """
    Modified data preprocessor class to work on input data.
    Certain 
    """

    def __init__(self):
        self.load_encoders()

    def load_encoders(self, dir: Path = "preprocess"):
        current_path = os.path.join(os.getcwd(), dir)
        file_names = [
            "pca.pkl",
            "minmax_scaler.pkl",
        ]
        for file_name in file_names:
            file_path = os.path.join(current_path, file_name)
            try:
                with open(file_path, 'rb') as file:
                    setattr(self, file_name.split('.')[0], pickle.load(file))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                setattr(self, file_name.split('.')[0], None)
            except pickle.UnpicklingError:
                print(f"Error: Failed to unpickle file: {file_path}")
                setattr(self, file_name.split('.')[0], None)

    def perform_pca(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Performs PCA on the input df.
        """
        return self.pca.transform(df)

    def get_df(self)->pd.DataFrame:
        return self.df
    

    def __validate_df(self):
        """
        Validate that df is not empty and all columns are in place
        """
        return True
    
    def preprocess_input(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Perform all preprocessing steps.
        """
        # Doing this for PCA
        scalar_values = df[["tenure_months", "total_long_distance_fee", "total_charges_quarter", "num_dependents", "num_referrals" ]]
        to_map = df[["contract_type", "has_premium_tech_support", "married", "has_device_protection", "has_online_backup"]]
        # Using loaded scaler to transform 
        new_df = pd.DataFrame(self.minmax_scaler.transform(scalar_values), columns=scalar_values.columns)
        for column in to_map.columns:
            to_map[column] = self.__map_categorical(to_map[column], 
                                                    custom_mapping={
                                                        'Month-to-Month':0, 
                                                        'One Year':1, 
                                                        'Two Year':2,
                                                        "Yes":1,
                                                        "No":0})   
        new_df = pd.concat([new_df, to_map], axis=1)    

        self.df = self.perform_pca(new_df)
        
        return self.df

    
    def __label_encode(self, col:pd.Series, on_input:bool=False)->pd.Series:
        if on_input:
            return self.label_encoder.transform(col)
        return self.label_encoder.fit_transform(col)        

    
    def __map_categorical(self, col: pd.Series, 
                          custom_mapping:Optional[dict]=None, 
                          on_input:bool=False
                          )->pd.Series:
        """
        Map common values to numerical values. Provide custom mapping if required.
        ---
        Arguments
        - on_input: If true, will be using the self.object to apply
        """
        if on_input:
            return col.map(self.mapping)
        self.mapping = {
            "Yes": 1,
            "No": 0,
            "Male": 1,
            "Female": 0,
            "Joined": 0,
            "Stayed": 1,
            "Churned": 2,
        }
        if custom_mapping:
            return col.map(custom_mapping)
        return col.map(self.mapping)
    
    def __binary_encode(self, col:pd.Series, on_input:bool=False)->pd.Series:
        """
        Returns a pd.Df that has to be concatted to the main df using the following:
        output_df = pd.concat([output_df, return_value], axis=1)
        ---
        Arguments
        - on_input: If true, will be using the self.object to apply
        """
        if on_input:
            return self.binary_encoder.transform(col)
        self.binary_encoder = BinaryEncoder(cols=['churn_category'])
        self.binary_encoder.fit_transform(col)
        churn_cat = self.binary_encoder.transform(col)
        return churn_cat
    
    def __scale_numerical(self,col:pd.Series, on_input:bool=False)->pd.Series:
        """
        Scales numerical columns using the same scaler object associated with this class.
        """
        if on_input:
            return self.scaler.transform(col.values.reshape(-1, 1))
        scaled_col = self.scaler.fit_transform(col.values.reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_col, columns=[col.name])
        return scaled_df
    
    def __one_hot_encode(self, col: pd.Series, one_hot_encoder:None, on_input:bool=False) -> pd.DataFrame:
        """
        One hot encodes a categorical column using the same encoder object associated with this class.
        Returns the result as a DataFrame.
        """
        if on_input:
           col_reshaped = col.values.reshape(-1, 1)
           csr_ohe_features = one_hot_encoder.transform(col_reshaped) 
           ohe_df = pd.DataFrame.sparse.from_spmatrix(csr_ohe_features)
           ohe_df.columns = one_hot_encoder.categories_[0]
           return ohe_df
        
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit(col.values.reshape(-1, 1))
        csr_ohe_features = one_hot_encoder.transform(col.values.reshape(-1, 1))
        ohe_df = pd.DataFrame.sparse.from_spmatrix(csr_ohe_features)

        # Assign column names based on the encoder categories
        ohe_df.columns = one_hot_encoder.categories_[0]

        with open(f'backend/preprocess/{col.name}_ohe.pkl', 'wb') as file:
            pickle.dump(one_hot_encoder, file)

        return ohe_df
    
if __name__ == "__main__":
    # Building ETL Object
    # etl = DataETL()
    # etl.use_local_data("research/data_given/")
    # etl.join_tables()

    dp = DataPreprocessor()
    output = dp.preprocess_input(df)
 
 

