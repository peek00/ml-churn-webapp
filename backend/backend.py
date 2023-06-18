from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import pickle
from pathlib import Path
import os
import pandas as pd



app = Flask(__name__)

def load_encoders(dir: Path = "preprocess"):
    min_max_scaler_path = os.path.join(dir, "minmax_scaler.pkl")
    one_hot_encoder_path = os.path.join(dir, "one_hot_encoder.pkl")
    label_encoder_path = os.path.join(dir,"label_encoder.pkl")
    categorical_mapping_path = os.path.join(dir,"categorical_mapping.pkl")
    pca_path = os.path.join(dir,"pca.pkl")
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
        with open(one_hot_encoder_path, 'rb') as file:
            one_hot_encoder = pickle.load(file)
    except FileNotFoundError:
        print(f"One-hot encoder file not found: {one_hot_encoder_path}")
        one_hot_encoder = None
    except pickle.UnpicklingError:
        print(f"Error: Failed to unpickle one-hot encoder file: {one_hot_encoder_path}")
        one_hot_encoder = None

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

    return min_max_scaler, one_hot_encoder, label_encoder, categorical_mapping, pca

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


def preprocess_input(data:dict):
    try:
        # Load encoders
        min_max_scaler, one_hot_encoder, label_encoder, categorical_mapping, pca = load_encoders()
        # Convert data from JSON into panda.df
        input_df = pd.DataFrame.from_dict(data, orient='index').transpose()

        # Preprocess categorical data
        input_df = preprocess_categorical(input_df, categorical_mapping)
        print(input_df.head())

        # Drop cols if exists
        columns_to_drop = ['status', 'customer_id', 'account_id', 'zip_code']
        existing_columns = list(filter(lambda col: col in data.columns, columns_to_drop))
        print(existing_columns)
        if existing_columns:
            features = data.drop(columns=existing_columns)
            # Load pca

        else:
            features = data.copy()
            return input_df
    

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

