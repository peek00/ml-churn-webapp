from flask import Flask, request, jsonify

app = Flask(__name__)

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
def get_predictions():
    # Extract data from the request
    request_data = request.get_json()

    # Your logic to process the data
    message = request_data.get('message', '')
    processed_data = {'processed_message': message}

    return jsonify(processed_data)

if __name__ == '__main__':
    # Enable hot reloading
    app.debug = True
    app.run()

