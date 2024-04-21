from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('crop_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    input_data = pd.DataFrame(data)

    # Make prediction
    recommended_crop = model.predict(input_data)

    # Prepare response
    response = {
        "recommended_crop": recommended_crop[0]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
