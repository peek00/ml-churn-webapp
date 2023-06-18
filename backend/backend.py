from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from category_encoders import BinaryEncoder
import pickle
from pathlib import Path
import os
import pandas as pd

from backend_data_preprocessor import DataPreprocessor


app = Flask(__name__)

def load_pca(dir: Path = "preprocess"):
    pca_path = os.path.join(dir,"pca.pkl")
    try:
        with open(pca_path, 'rb') as file:
            pca = pickle.load(file)
    except FileNotFoundError:
        print(f"Pca file not found: {pca_path}")
        pca = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle pca mapping file: {pca_path}")
        pca = None

    return pca

def preprocess_categorical(df:pd.DataFrame, mapping:dict)-> pd.DataFrame:
    expected_columns = [
        'has_internet_service', 'has_phone_service', 'has_unlimited_data',
        'has_multiple_lines', 'has_premium_tech_support', 'has_online_security',
        'has_online_backup', 'has_device_protection', 'paperless_billing',
        'stream_movie', 'stream_music', 'stream_tv', 'senior_citizen',
        'married', 'gender', 'contract_type'
    ]

    for column in expected_columns:
        assert column in df.columns, f"Column '{column}' does not exist in the DataFrame."

    for column in expected_columns:
        df[column] = df[column].map(mapping)
    return df

def preprocess_numerical(df:pd.DataFrame, scaler:MinMaxScaler)-> pd.DataFrame:
    expected_columns = [
        'num_referrals', 
        'age',
        'tenure_months',
        'avg_long_distance_fee_monthly',
        'total_long_distance_fee',
        'avg_gb_download_monthly',
        'total_monthly_fee',
        'total_charges_quarter',
        'total_refunds',
        'population'
    ]

    for column in expected_columns:
        assert column in df.columns, f"Column '{column}' does not exist in the DataFrame."
    
    for column in expected_columns:
        df[column] = scaler.transform(df[column].values.reshape(-1,1))

    return df

def preprocess_label_and_binary(df:pd.DataFrame, label_encoder:LabelEncoder, binary_encoder: BinaryEncoder)-> pd.DataFrame:
    # Label encoding
    assert "status" in df.columns, "Column 'status' does not exist in the DataFrame."
    df['status'] = label_encoder.transform(df['status'])
    # Binary encoding
    assert "churn_category" in df.columns, "Column 'churn_category' does not exist in the DataFrame."
    churn_cat = binary_encoder.transform(df['churn_category'])
    df.drop(columns=['churn_category'], inplace=True)
    df= pd.concat([df, churn_cat], axis=1)

    return df

def preprocess_one_hot_encoding(df:pd.DataFrame, col_name:str, ohe:OneHotEncoder)->pd.DataFrame:
    
    assert col_name in df.columns, f"Column '{col_name}' does not exist in the DataFrame."
    col_reshaped = df[col_name].values.reshape(-1, 1)
    csr_ohe_features = ohe.transform(col_reshaped)
    ohe_df = pd.DataFrame.sparse.from_spmatrix(csr_ohe_features)
    ohe_df.columns = ohe.categories_[0]
    df.drop(columns=[col_name], inplace=True)
    df = pd.concat([df, ohe_df], axis=1)
    return df

def preprocess_churn_labels(df:pd.DataFrame)->pd.DataFrame:
    # Fixing churn_labels
    df.loc[df['status'] == 2, 'churn_label'] = 1
    df.loc[df['status'] == 0, 'churn_label'] = 0
    df.loc[df['status'] == 1, 'churn_label'] = 0
    return df

def preprocess_input(data:dict, dir:Path="preprocess"):
    data_processer = DataPreprocessor()
    input_df = pd.DataFrame.from_dict(data, orient='index').transpose()
    parsed_input = data_processer.preprocess_input(input_df)
    # Doing PCA
    pca = load_pca(dir)
    cols_to_drop = [ 'churn_label', 'status','customer_id', 'account_id', 'zip_code']
    for column in cols_to_drop:
        assert column in parsed_input.columns, f"Column '{column}' does not exist in the DataFrame."
    parsed_input = parsed_input.drop(cols_to_drop, axis=1)

    # Fit pca
    transformed_features = pca.transform(parsed_input)
    print(transformed_features)
    return 1
    # try:
    #     # Load encoders
    #     min_max_scaler, internet_type_ohe, payment_method_ohe, label_encoder, categorical_mapping, binary_encoder, pca = load_encoders()
    #     # Convert data from JSON into panda.df
    #     input_df = pd.DataFrame.from_dict(data, orient='index').transpose()

    #     # Preprocess categorical data
    #     input_df = preprocess_categorical(input_df, categorical_mapping)
    #     input_df = preprocess_numerical(input_df, min_max_scaler)
    #     input_df = preprocess_label_and_binary(input_df, label_encoder, binary_encoder)
    #     input_df = preprocess_one_hot_encoding(input_df, "internet_type", internet_type_ohe)
    #     input_df = preprocess_one_hot_encoding(input_df, "payment_method", payment_method_ohe)
    #     input_df = preprocess_churn_labels(input_df)

    #     # Dropping
    #     cols_to_drop = ['churn_reason', 'churn_label', 'city', 'latitutde', 'longitude', 'area_id', 'status','customer_id', 'account_id', 'zip_code']
    #     for column in cols_to_drop:
    #         assert column in input_df.columns, f"Column '{column}' does not exist in the DataFrame."
    #     input_df = input_df.drop(cols_to_drop, axis=1)
    #     # input_df[float('nan')] = input_df['nan'].astype(str)
    #     # input_df.drop(columns=[float('nan')], inplace=True)
    #     input_df.columns = input_df.columns.astype(str)
    #     input_df.drop(columns=['nan'], inplace=True)

    #     for _ in input_df.columns:
    #         print(type(_), _)
    #     print(input_df.head())
    #     # Fit pca
    #     transformed_features = pca.transform(input_df)
    #     print(transformed_features.head())


    

    # except Exception as e:
    #     print(e)
    #     print("Failed to load encoders!")
    
    
    # # Your preprocessing logic goes here
    
    # return data

# GET endpoint
@app.route('/',  methods=['GET'])
def home():
    # Your logic to retrieve data
    data = {'message': 'This is a GET request'}
    return jsonify(data)

# GET endpoint
@app.route('/api/data', methods=['GET'])
def get_data():
    # Your logic to retrieve data
    data = {'message': 'This is a GET request'}
    return jsonify(data)

# POST endpoint
@app.route('/predict', methods=['POST'])
def get_prediction():

    # Load the saved model
    model_path = "model/model.pkl"
    model = joblib.load(model_path)

    # Get the data from the request
    data = request.get_json()
    preprocess_input(data)

    # Make predictions using the loaded model
    # predictions = model.predict(input_data)

    # Return the predictions as a JSON response
    return "test"
    return jsonify(predictions=predictions.tolist())

if __name__ == '__main__':
    # Enable hot reloading
    app.debug = True
    app.run()

