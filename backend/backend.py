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

def preprocess_input(data:dict, dir:Path="preprocess"):
    """
    Uses backend preprocess class to preprocess input data.
    """
    data_processer = DataPreprocessor()
    input_df = pd.DataFrame.from_dict(data, orient='index').transpose()
    parsed_input = data_processer.preprocess_input(input_df)
    # Doing PCA
    pca = load_pca(dir)
    cols_to_drop = [ 'zip_code']
    parsed_input = parsed_input.drop(cols_to_drop, axis=1)

    # Fit pca
    print(f"Processed order is {parsed_input.columns}")
    print(parsed_input.columns[parsed_input.isna().any()].tolist())

    transformed_features = pca.transform(parsed_input)
    return transformed_features


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
    # Load the saved model
    model_path = "model/model.pkl"
    model = joblib.load(model_path)

    # Get the data from the request
    data = {}
    for key, value in request.form.items():
        data[key] = value


    # Use the JSON dictionary for prediction or further processing
    
    processed_input = preprocess_input(data)
    

    # Load model
    model_path = "model/model.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Test values for prediction == 0
    # values = [[-1.240957, -0.914264, 0.437043, -0.800090, 0.421031]]
    # processed_input = np.array(values)
    predictions = model.predict(processed_input)

    return {
        "prediction": int(predictions[0].astype(int))
    }
    # Make predictions using the loaded model
    # predictions = model.predict(input_data)

    # Return the predictions as a JSON response
    return "test"
    return jsonify(predictions=predictions.tolist())

if __name__ == '__main__':
    # Enable hot reloading
    app.debug = True
    app.run()

