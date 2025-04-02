import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
with open("smoking_cessation_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([np.array(data)])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
