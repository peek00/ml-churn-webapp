import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from typing import Optional

from data_etl import DataETL

class DataPreprocessor:
    """
    Perform data cleaning steps on dataset.
    """

    def __init__(self, df:pd.DataFrame):
        """
        Assume that the table has been fully joined.
        """
        self.df = df
        print(self.df.keys())
        self.scaler = MinMaxScaler()
        self.one_hot_encoder = OneHotEncoder()
        self.label_encoder = LabelEncoder()
        # self._validate_df()
        self.__preprocess()

    def __validate_df(self):
        """
        Validate that df is not empty and all columns are in place
        """
        return True
    
    def __preprocess(self):
        """
        Perform all preprocessing steps.
        """
        # Map string values to categorical
        self.df['has_internet_service'] = self.__map_categorical(self.df['has_internet_service'])
        self.df['has_phone_service'] = self.__map_categorical(self.df['has_phone_service'])
        self.df['has_unlimited_data'] = self.__map_categorical(self.df['has_unlimited_data'])
        self.df['has_multiple_lines'] = self.__map_categorical(self.df['has_multiple_lines'])
        self.df['has_premium_tech_support'] = self.__map_categorical(self.df['has_premium_tech_support'])
        self.df['has_online_security'] = self.__map_categorical(self.df['has_online_security'])
        self.df['has_online_backup'] = self.__map_categorical(self.df['has_online_backup'])
        self.df['has_device_protection'] = self.__map_categorical(self.df['has_device_protection'])
        self.df['paperless_billing'] = self.__map_categorical(self.df['paperless_billing'])
        self.df['stream_movie'] = self.__map_categorical(self.df['stream_movie'])
        self.df['stream_music'] = self.__map_categorical(self.df['stream_music'])
        self.df['stream_tv'] = self.__map_categorical(self.df['stream_tv'])
        self.df['senior_citizen'] = self.__map_categorical(self.df['senior_citizen'])
        self.df['married'] = self.__map_categorical(self.df['married'])
        self.df['gender'] = self.__map_categorical(self.df['gender'])

        # Scaling numerical values
        self.df['num_referrals'] = self.__scale_numerical(self.df['num_referrals'])
        self.df['age'] = self.__scale_numerical(self.df['age'])
        self.df['tenure_months'] = self.__scale_numerical(self.df['tenure_months'])
        self.df['avg_long_distance_fee_monthly'] = self.__scale_numerical(self.df['avg_long_distance_fee_monthly'])
        self.df['total_long_distance_fee'] = self.__scale_numerical(self.df['total_long_distance_fee'])
        self.df['avg_gb_download_monthly'] = self.__scale_numerical(self.df['avg_gb_download_monthly'])
        self.df['total_monthly_fee'] = self.__scale_numerical(self.df['total_monthly_fee'])
        self.df['total_charges_quarter'] = self.__scale_numerical(self.df['total_charges_quarter'])
        self.df['total_refunds'] = self.__scale_numerical(self.df['total_refunds'])
        self.df['population'] = self.__scale_numerical(self.df['population'])

        # Label encode 
        self.df['status'] = self.__label_encode(self.df['status'])

        # Binary encode
        churn_df = self.__binary_encode(self.df['churn_category'])
        assert not churn_df.empty, "Churn DF is empty!"
        self.df = pd.concat([self.df, churn_df], axis=1)
        self.df.drop('churn_category', axis=1, inplace=True)

        # One hot encode
        print("Trying to one hot encode")
        print(f"Input is {self.df['payment_method']} ")
        _ = self.__one_hot_encode(self.df['payment_method'])
        print(f"Output is {_} ")





    def __label_encode(self, col:pd.Series)->pd.Series:
        return self.label_encoder.fit_transform(col)        

    
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
        scaled_col = self.scaler.fit_transform(col.values.reshape(-1, 1))
        scaled_df = pd.DataFrame(scaled_col, columns=[col.name])
        return scaled_df
    
    def __one_hot_encode(self, col: pd.Series) -> pd.DataFrame:
        """
        One hot encodes a categorical column using the same encoder object associated with this class.
        Returns the result as a DataFrame.
        """
        oh_encoded = pd.get_dummies(col)
        return oh_encoded

        
    
if __name__ == "__main__":
    # Building ETL Object
    etl = DataETL()
    etl.use_local_data("research/data_given/")
    etl.join_tables()

    df = etl.get_df()

    dp = DataPreprocessor(df)
    dp.df.head()
 

