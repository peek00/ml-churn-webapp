import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from typing import Optional
from pathlib import Path
import pickle
import os


class DataPreprocessor:
    """
    Modified data preprocessor class to work on input data.
    Certain 
    """

    def __init__(self):
        """
        Assume that the table has been fully joined.
        """
        self.load_encoders()

    def load_encoders(self, dir: Path = "preprocess"):
        current_path = os.path.join(os.getcwd(), dir)
        file_names = [
            "mapping.pkl",
            # "internet_type_ohe.pkl",
            # "label_encoder.pkl",
            "minmax_scaler.pkl",
            # "payment_method_ohe.pkl"
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

    def get_df(self)->pd.DataFrame:
        return self.df
    

    def __validate_df(self):
        """
        Validate that df is not empty and all columns are in place
        """
        return True
    
    def preprocess_input(self, df:pd.DataFrame, model:str)->pd.DataFrame:
        """
        Perform all preprocessing steps.
        """
        print(df.head())
        if model == "catboost":
            # Transform using self.minmaxscalar
            new_df_array = self.minmax_scaler.transform(df[["tenure_months", "total_charges_quarter", "num_referrals", "total_long_distance_fee"]])
            new_df = pd.DataFrame(new_df_array, columns=["tenure_months", "total_charges_quarter", "num_referrals", "total_long_distance_fee"])
            new_df['contract_type'] = self.__map_categorical(df['contract_type'], custom_mapping={'Month-to-Month':0, 'One Year':1, 'Two Year':2})
            
            self.df = new_df
            return self.df
        elif model == "xgboost":
            # Map string values to categorical
            self.df = df
            assert "has_internet_service" in self.df.columns, "has_internet_service column not found"
            self.df['has_internet_service'] = self.__map_categorical(self.df['has_internet_service'], on_input=True)
            self.df['has_phone_service'] = self.__map_categorical(self.df['has_phone_service'], on_input=True)
            self.df['has_unlimited_data'] = self.__map_categorical(self.df['has_unlimited_data'], on_input=True)
            self.df['has_multiple_lines'] = self.__map_categorical(self.df['has_multiple_lines'], on_input=True)
            self.df['has_premium_tech_support'] = self.__map_categorical(self.df['has_premium_tech_support'], on_input=True)
            self.df['has_online_security'] = self.__map_categorical(self.df['has_online_security'], on_input=True)
            self.df['has_online_backup'] = self.__map_categorical(self.df['has_online_backup'], on_input=True)
            self.df['has_device_protection'] = self.__map_categorical(self.df['has_device_protection'], on_input=True)
            self.df['paperless_billing'] = self.__map_categorical(self.df['paperless_billing'], on_input=True)
            self.df['stream_movie'] = self.__map_categorical(self.df['stream_movie'], on_input=True)
            self.df['stream_music'] = self.__map_categorical(self.df['stream_music'], on_input=True)
            self.df['stream_tv'] = self.__map_categorical(self.df['stream_tv'], on_input=True)
            self.df['senior_citizen'] = self.__map_categorical(self.df['senior_citizen'], on_input=True)
            self.df['married'] = self.__map_categorical(self.df['married'], on_input=True)
            self.df['gender'] = self.__map_categorical(self.df['gender'], on_input=True)
            self.df['contract_type'] = self.__map_categorical(self.df['contract_type'], custom_mapping={'Month-to-Month':0, 'One Year':1, 'Two Year':2})

            # Scaling numerical values
            self.df['num_referrals'] = self.__scale_numerical(self.df['num_referrals'], on_input=True)
            self.df['age'] = self.__scale_numerical(self.df['age'], on_input=True)
            self.df['tenure_months'] = self.__scale_numerical(self.df['tenure_months'], on_input=True)
            self.df['avg_long_distance_fee_monthly'] = self.__scale_numerical(self.df['avg_long_distance_fee_monthly'], on_input=True)
            self.df['total_long_distance_fee'] = self.__scale_numerical(self.df['total_long_distance_fee'], on_input=True)
            self.df['avg_gb_download_monthly'] = self.__scale_numerical(self.df['avg_gb_download_monthly'], on_input=True)
            self.df['total_monthly_fee'] = self.__scale_numerical(self.df['total_monthly_fee'], on_input=True)
            self.df['total_charges_quarter'] = self.__scale_numerical(self.df['total_charges_quarter'], on_input=True)
            self.df['total_refunds'] = self.__scale_numerical(self.df['total_refunds'], on_input=True)
            self.df['population'] = self.__scale_numerical(self.df['population'], on_input=True)

            # One hot encode
            # Payment method
            ohe_payment_method = self.__one_hot_encode(self.df['payment_method'], self.payment_method_ohe, on_input=True)
            for col_name in ohe_payment_method.columns:
                assert col_name not in self.df.columns, f"Column '{col_name}' already exists in the DataFrame."
            self.df = pd.concat([self.df, ohe_payment_method], axis=1)
            # Dropping payment method
            self.df = self.df.drop('payment_method', axis=1)
            # Internet_type
            ohe_internet_type = self.__one_hot_encode(self.df['internet_type'], self.internet_type_ohe, on_input=True)
            for col_name in ohe_internet_type.columns:
                assert col_name not in self.df.columns, f"Column '{col_name}' already exists in the DataFrame."

            self.df = pd.concat([self.df, ohe_internet_type], axis=1)
            # Dropping internet type
            self.df = self.df.drop('internet_type', axis=1)
            self.df.columns = self.df.columns.astype(str)

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
 
 

