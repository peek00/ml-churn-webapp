from flask import Flask, request, jsonify, render_template
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
    """
    Loads PCA object to transform input data for model.
    """
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

def preprocess_input(data:dict, dir:Path="preprocess", model="catboost"):
    """
    Uses backend preprocess class to preprocess input data.
    """
    data_processer = DataPreprocessor()
    input_df = pd.DataFrame.from_dict(data, orient='index').transpose()
    parsed_input = data_processer.preprocess_input(input_df, model)
    # Doing PCA
    if model =="xgboost":
        pca = load_pca(dir)
        cols_to_drop = [ 'zip_code']
        parsed_input = parsed_input.drop(cols_to_drop, axis=1)
        # Fit pca
        transformed_features = pca.transform(parsed_input)
        return transformed_features
    if model == "catboost":
        features_list = ['contract_type', 'tenure_months', 'total_long_distance_fee', 'total_charges_quarter', 'num_referrals' ]
        parsed_input = parsed_input[features_list]
        return parsed_input

# GET endpoint
@app.route('/',  methods=['GET'])
def home():
    """
    Form page
    """
    data = {'message': 'This is a GET request'}
    return render_template('index.html')

# GET endpoint
@app.route('/api/data', methods=['GET'])
def get_data():
    data = {'message': 'This is a GET request'}
    return jsonify(data)

# POST endpoint
@app.route('/predict', methods=['POST'])
def get_prediction():
    """
    Redirected here with information from the form.
    """
    # Get the data from the request
    data = {}
    if request.form:
        input_data = request.form.items()
    else:
        # Assuming data is sent as JSON
        input_data = request.get_json().items()
    
    for key, value in input_data:
        data[key] = value
    # Use the JSON dictionary for prediction or further processing
    processed_input = preprocess_input(data, model="catboost")
    # Load model
    model_path = "model/catboost_model.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    # Test values for prediction == 0
    # values = [[-1.240957, -0.914264, 0.437043, -0.800090, 0.421031]]
    # processed_input = np.array(values)
    predictions = model.predict(processed_input)
    return {
        "prediction": int(predictions[0].astype(int))
    }


if __name__ == '__main__':
    # Enable hot reloading
    app.debug = True
    app.run()

