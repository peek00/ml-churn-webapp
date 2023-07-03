from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
CORS(app)

def load_encoders(dir: Path = "preprocess"):
    """
    Loads PCA and minmax scaler object to transform input data for model.
    """
    encoders = ["pca.pkl", "minmax_scaler.pkl"]
    objects = []
    for encoder in encoders:
        path = os.path.join(dir, encoder)
        try:
            with open(path, 'rb') as file:
                item = pickle.load(file)
                objects.append(item)
        except FileNotFoundError:
            print(f"{encoder} file not found: {path}")
            
        except pickle.UnpicklingError:
            print(f"Error: Failed to unpickle {encoder} file: {path}")
    return objects

def preprocess_input(data:dict, dir:Path="preprocess"):
    """
    Uses backend preprocess class to preprocess input data.
    """
    data_processer = DataPreprocessor()
    input_df = pd.DataFrame.from_dict(data, orient='index').transpose()
    parsed_input = data_processer.preprocess_input(input_df)
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
        if key in ['tenure_months', 'num_referrals', 'total_long_distance_fee', 'total_charges_quarter']:
            data[key]= float(value)
        else:
            data[key] = value
    # Use the JSON dictionary for prediction or further processing
    processed_input = preprocess_input(data)
    # Load model
    model_path = "model/catboost_model.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    # Test values for prediction == 0
    # values = [[-1.240957, -0.914264, 0.437043, -0.800090, 0.421031]]
    # processed_input = np.array(values)
    binary_predict = model.predict(processed_input)
    logits = model.predict_proba(processed_input)
    return {
        "prediction": int(binary_predict[0].astype(int)),
        "logits": logits.astype(float).tolist(), 
        "processed_input": processed_input.tolist(),
    }


if __name__ == '__main__':
    # Enable hot reloading
    app.debug = True
    app.run()

