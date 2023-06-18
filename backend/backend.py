from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from category_encoders import BinaryEncoder
import pickle
from pathlib import Path
import os
import pandas as pd



app = Flask(__name__)

def load_encoders(dir: Path = "preprocess"):
    min_max_scaler_path = os.path.join(dir, "minmax_scaler.pkl")
    label_encoder_path = os.path.join(dir,"label_encoder.pkl")
    categorical_mapping_path = os.path.join(dir,"categorical_mapping.pkl")
    pca_path = os.path.join(dir,"pca.pkl")
    binary_encoder_path = os.path.join(dir,"binary_encoder.pkl")

    internet_type_ohe_path = os.path.join(dir,"internet_type_ohe.pkl")
    payment_method_ohe_path = os.path.join(dir,"payment_method_ohe.pkl")
    try:
        with open(min_max_scaler_path, 'rb') as file:
            min_max_scaler = pickle.load(file)
    except FileNotFoundError:
        print(f"Min-max scaler file not found: {min_max_scaler_path}")
        min_max_scaler = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle min-max scaler file: {min_max_scaler_path}")
        min_max_scaler = None

    try:
        with open(internet_type_ohe_path, 'rb') as file:
            internet_type_ohe = pickle.load(file)
    except FileNotFoundError:
        print(f"One-hot encoder file not found: {internet_type_ohe_path}")
        internet_type_ohe = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle one-hot encoder file: {internet_type_ohe_path}")
        internet_type_ohe = None

    try:
        with open(payment_method_ohe_path, 'rb') as file:
            payment_method_ohe = pickle.load(file)
    except FileNotFoundError:
        print(f"One-hot encoder file not found: {payment_method_ohe_path}")
        payment_method_ohe = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle one-hot encoder file: {payment_method_ohe_path}")
        payment_method_ohe = None

    try:
        with open(label_encoder_path, 'rb') as file:
            label_encoder = pickle.load(file)
    except FileNotFoundError:
        print(f"Label encoder file not found: {label_encoder_path}")
        label_encoder = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle label encoder file: {label_encoder_path}")
        label_encoder = None

    try:
        with open(categorical_mapping_path, 'rb') as file:
            categorical_mapping = pickle.load(file)
    except FileNotFoundError:
        print(f"Categorical mapping file not found: {categorical_mapping_path}")
        categorical_mapping = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle categorical mapping file: {categorical_mapping_path}")
        categorical_mapping = None

    try:
        with open(pca_path, 'rb') as file:
            pca = pickle.load(file)
    except FileNotFoundError:
        print(f"Pca file not found: {pca_path}")
        pca = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle pca mapping file: {pca_path}")
        pca = None

    try:
        with open(binary_encoder_path, 'rb') as file:
            binary_encoder = pickle.load(file)
    except FileNotFoundError:
        print(f"Binary_encoder file not found: {binary_encoder_path}")
        binary_encoder = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle binary_encoder mapping file: {binary_encoder_path}")
        binary_encoder = None

    return min_max_scaler, internet_type_ohe, payment_method_ohe , label_encoder, categorical_mapping, binary_encoder, pca

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
    for column in expected_columns:
            # Reshape input data to 2D format if needed
        col_reshaped = df[column].values.reshape(-1, 1)

        # Use the fitted OneHotEncoder object to transform the data
        csr_ohe_features = ohe.transform(col_reshaped)
        ohe_df = pd.DataFrame.sparse.from_spmatrix(csr_ohe_features)

        # Assign column names based on the encoder categories
        ohe_df.columns = ohe.categories_[0]
        print(ohe_df.head())
        print(column)
        print(df)
        print(df[column].head())
        df.drop(columns=[column], inplace=True)
        # df = pd.concat([df, ohe_output], axis=1)
    
    return df



def preprocess_input(data:dict):
    try:
        # Load encoders
        min_max_scaler, internet_type_ohe, payment_method_ohe, label_encoder, categorical_mapping, binary_encoder, pca = load_encoders()
        # Convert data from JSON into panda.df
        input_df = pd.DataFrame.from_dict(data, orient='index').transpose()

        # Preprocess categorical data
        input_df = preprocess_categorical(input_df, categorical_mapping)
        input_df = preprocess_numerical(input_df, min_max_scaler)
        input_df = preprocess_label_and_binary(input_df, label_encoder, binary_encoder)
        print("Error here")
        input_df = preprocess_one_hot_encoding(input_df, "internet_type", internet_type_ohe)
        input_df = preprocess_one_hot_encoding(input_df, "payment_method", payment_method_ohe)
        print(input_df.head())


    

    except Exception as e:
        print(e)
        print("Failed to load encoders!")
    
    
    # Your preprocessing logic goes here
    
    return data

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

