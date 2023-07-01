import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from typing import Optional
from pathlib import Path
import pickle
from typing import List
import os

from data_etl import DataETL

class DataPreprocessor:
    """
    Perform data cleaning steps on dataset.
    """

    def __init__(self, df:pd.DataFrame):
        """
        Takes in a fully joined dataframe and a list of encoders to use.
        """
        self.df = df
        self.scaler = MinMaxScaler()
        self.one_hot_encoder = OneHotEncoder()
        self.label_encoder = LabelEncoder()
        # self._validate_df()
        # self.__preprocess()
        
        # pd.set_option('display.max_columns', None)  # None will display all columns
        # print(self.df.head())

    def preprocess(self):
        self.__preprocess()
    
    def __load_encoder(self, encoder_name:str, dir:Path="preprocess"):
        """
        Assuming files are in a folder called preprocess and are labelled as name.pkl
        """
        encoder_filename = f"{encoder_name}.pkl"
        encoder_path = os.path.join(dir, encoder_filename)
        try:
            with open(encoder_path, 'rb') as file:
                setattr(self, encoder_name, pickle.load(file))
        except FileNotFoundError:
            print(f"Encoder file not found: {encoder_path}")
            encoder = None
        except pickle.UnpicklingError:
            print(f"Error: Failed to unpickle encoder file: {encoder_path}")
            encoder = None


    def transform(self, target:str, features:List[str], encoders:List[str]):
        """
        This transform is done for the training phase.
        """
        
        new_df = self.df[features]
        new_df[target] = self.df[target]

        # Loops through encoders and applies them
        for encoder in encoders:
            for encoder_name, features in encoder.items():
                self.__load_encoder(encoder_name)
                new_df[features] = getattr(self, encoder_name).transform(new_df[features])
        
        # Specific to this, load in target and customer status to preprocess them
        mapping = {
            "Joined": 0,
            "Stayed": 1,
            "Churned": 2,
            'Month-to-Month':0, 
            'One Year':1, 
            'Two Year':2
        }

        # Preprocessing the target
        target_df = pd.DataFrame()
        target_df[['churn_label', 'customer_status']] = self.df[['churn_label', 'customer_status']]
        target_df['customer_status'] = target_df['customer_status'].map(mapping)
        target_df.loc[target_df['customer_status'] == 2, 'churn_label'] = 1
        target_df.loc[target_df['customer_status'] == 0, 'churn_label'] = 0
        target_df.loc[target_df['customer_status'] == 1, 'churn_label'] = 0

        new_df['churn_label'] = target_df['churn_label']
        new_df['contract_type'] = new_df['contract_type'].map(mapping)

        self.transformed_df = new_df
        return self.transformed_df
    
    def get_df(self)->pd.DataFrame:
        return self.df
    

    def __validate_df(self):
        """
        Validate that df is not empty and all columns are in place
        """
        return True
    
    def __preprocess(self):
        """
        Perform all preprocessing steps.
        """
        # Doing this for catboost ONLY
        # new_df = pd.DataFrame()
        # new_df['contract_type'] = self.__map_categorical(self.df['contract_type'], custom_mapping={'Month-to-Month':0, 'One Year':1, 'Two Year':2})
        # new_df['num_referrals'] = self.df['num_referrals']
        # new_df['tenure_months'] = self.df['tenure_months']
        # new_df['total_long_distance_fee'] = self.df['total_long_distance_fee']
        # new_df['total_charges_quarter'] = self.df['total_charges_quarter']
        # new_df['customer_status'] = self.df['customer_status']
        # new_df['churn_label'] = self.df['churn_label']
        # new_df['customer_status'] = self.__label_encode(self.df['customer_status'])
        #  # Fixing churn_labels
        # new_df.loc[new_df['customer_status'] == 2, 'churn_label'] = 1
        # new_df.loc[new_df['customer_status'] == 0, 'churn_label'] = 0
        # new_df.loc[new_df['customer_status'] == 1, 'churn_label'] = 0

        # Doing this for PCA
        scalar_values = self.df[["tenure_months", "total_long_distance_fee", "total_charges_quarter", "num_dependents" ]]
        to_map = self.df[["contract_type", "has_premium_tech_support", "married", "has_device_protection", "has_online_backup"]]

        self.scaler = MinMaxScaler()
        self.scaler.fit(scalar_values)
        new_df = pd.DataFrame(self.scaler.transform(scalar_values), columns=scalar_values.columns)
        for column in to_map.columns:
            to_map[column] = self.__map_categorical(to_map[column], 
                                                    custom_mapping={
                                                        'Month-to-Month':0, 
                                                        'One Year':1, 
                                                        'Two Year':2,
                                                        "Yes":1,
                                                        "No":0})   
        new_df = pd.concat([new_df, to_map], axis=1)    

        new_df['customer_status'] = self.__label_encode(self.df['customer_status'])
         # Fixing churn_labels
        new_df.loc[new_df['customer_status'] == 2, 'churn_label'] = 1
        new_df.loc[new_df['customer_status'] == 0, 'churn_label'] = 0
        new_df.loc[new_df['customer_status'] == 1, 'churn_label'] = 0
        new_df.drop('customer_status', axis=1, inplace=True)
        print(new_df.head())
        self.df = new_df

        # Creating a new min max scalar
        # self.scaler = MinMaxScaler()
        # self.scaler.fit(new_df[['tenure_months', 'total_charges_quarter', "num_referrals", "total_long_distance_fee"]])
        # new_df[['tenure_months', 'total_charges_quarter', "num_referrals", "total_long_distance_fee"]] = self.scaler.transform(new_df[['tenure_months', 'total_charges_quarter', "num_referrals", "total_long_distance_fee"]])
        # new_df.drop('customer_status', axis=1, inplace=True)
        # print(new_df.head())
        # self.df = new_df


        # # Map string values to categorical
        # self.df['has_internet_service'] = self.__map_categorical(self.df['has_internet_service'])
        # self.df['has_phone_service'] = self.__map_categorical(self.df['has_phone_service'])
        # self.df['has_unlimited_data'] = self.__map_categorical(self.df['has_unlimited_data'])
        # self.df['has_multiple_lines'] = self.__map_categorical(self.df['has_multiple_lines'])
        # self.df['has_premium_tech_support'] = self.__map_categorical(self.df['has_premium_tech_support'])
        # self.df['has_online_security'] = self.__map_categorical(self.df['has_online_security'])
        # self.df['has_online_backup'] = self.__map_categorical(self.df['has_online_backup'])
        # self.df['has_device_protection'] = self.__map_categorical(self.df['has_device_protection'])
        # self.df['paperless_billing'] = self.__map_categorical(self.df['paperless_billing'])
        # self.df['stream_movie'] = self.__map_categorical(self.df['stream_movie'])
        # self.df['stream_music'] = self.__map_categorical(self.df['stream_music'])
        # self.df['stream_tv'] = self.__map_categorical(self.df['stream_tv'])
        # self.df['senior_citizen'] = self.__map_categorical(self.df['senior_citizen'])
        # self.df['married'] = self.__map_categorical(self.df['married'])
        # self.df['gender'] = self.__map_categorical(self.df['gender'])
        # self.df['contract_type'] = self.__map_categorical(self.df['contract_type'], custom_mapping={'Month-to-Month':0, 'One Year':1, 'Two Year':2})

        # # Scaling numerical values
        # self.df['num_referrals'] = self.__scale_numerical(self.df['num_referrals'])
        # self.df['age'] = self.__scale_numerical(self.df['age'])
        # self.df['tenure_months'] = self.__scale_numerical(self.df['tenure_months'])
        # self.df['avg_long_distance_fee_monthly'] = self.__scale_numerical(self.df['avg_long_distance_fee_monthly'])
        # self.df['total_long_distance_fee'] = self.__scale_numerical(self.df['total_long_distance_fee'])
        # self.df['avg_gb_download_monthly'] = self.__scale_numerical(self.df['avg_gb_download_monthly'])
        # self.df['total_monthly_fee'] = self.__scale_numerical(self.df['total_monthly_fee'])
        # self.df['total_charges_quarter'] = self.__scale_numerical(self.df['total_charges_quarter'])
        # self.df['total_refunds'] = self.__scale_numerical(self.df['total_refunds'])
        # # self.df['population'] = self.__scale_numerical(self.df['population'])

        # # Label encode 
        # self.df['customer_status'] = self.__label_encode(self.df['customer_status'])

        # # Binary encode
        # # churn_df = self.__binary_encode(self.df['churn_category'])
        # # assert not churn_df.empty, "Churn DF is empty!"
        # # self.df = pd.concat([self.df, churn_df], axis=1)
        # self.df.drop('churn_category', axis=1, inplace=True)

        # # One hot encode
        # # Payment method
        # one_hot_encoder = OneHotEncoder()
        # ohe_payment_method = self.__one_hot_encode(self.df['payment_method'], one_hot_encoder)
        # for col_name in ohe_payment_method.columns:
        #     assert col_name not in self.df.columns, f"Column '{col_name}' already exists in the DataFrame."
        # self.df = pd.concat([self.df, ohe_payment_method], axis=1)
        # # Dropping payment method
        # self.df = self.df.drop('payment_method', axis=1)
        # # Internet_type
        # one_hot_encoder = OneHotEncoder()
        # ohe_internet_type = self.__one_hot_encode(self.df['internet_type'], one_hot_encoder)
        # for col_name in ohe_internet_type.columns:
        #     assert col_name not in self.df.columns, f"Column '{col_name}' already exists in the DataFrame."

        # self.df = pd.concat([self.df, ohe_internet_type], axis=1)
        # # Dropping internet type
        # self.df = self.df.drop('internet_type', axis=1)
        # self.df.columns = self.df.columns.astype(str)


        # # Fixing churn_labels
        # self.df.loc[self.df['customer_status'] == 2, 'churn_label'] = 1
        # self.df.loc[self.df['customer_status'] == 0, 'churn_label'] = 0
        # self.df.loc[self.df['customer_status'] == 1, 'churn_label'] = 0

        # # Dropping
        # cols_to_drop = ['churn_reason', 'city_name', 'latitude', 'longitude', 'area_id']
        # self.df = self.df.drop(cols_to_drop, axis=1)

        # print(self.df['churn_label'].value_counts())
    
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
        
        # oh_encoded = pd.get_dummies(col)
        return oh_encoded
    
    def get_NaN_count(self)->dict:
        """
        Returns the number of NaN values for each column in a dictionary.
        """
        nan_count = self.df.isna().sum().to_dict()
        return nan_count
    
    def save(self):
        """
        Saves the PCA and scaling objects along with any other relevant information.
        """
        # Save PCA object
        # Save scaling object
        with open('backend/preprocess/minmax_scaler.pkl', 'wb') as file:
            pickle.dump(self.scaler, file)
        # with open('backend/preprocess/label_encoder.pkl', 'wb') as file:
        #     pickle.dump(self.label_encoder, file)
        # with open('backend/preprocess/categorical_mapping.pkl', 'wb') as file:
        #     pickle.dump(self.mapping, file)
        # with open('backend/preprocess/binary_encoder.pkl', 'wb') as file:
        #     pickle.dump(self.binary_encoder, file)
        print("Pickle objects!")

    
if __name__ == "__main__":
    # Building ETL Object
    etl = DataETL()
    # etl.use_local_data("research/data_given/")
    etl.retrieve_tables()
    etl.join_tables()

    df = etl.get_df()

    dp = DataPreprocessor(df)
    dp.df.head()
    dp.save()
 

