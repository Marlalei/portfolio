from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn import * 
app = Flask(__name__)

# Load the model, vectorizer, and label encoder
def load_model():
    model = joblib.load("model/marla-neo-001.pkl")
    vectorizer = joblib.load("model/vectorizer-neo1.pkl")
    label_encoder = joblib.load("model/label_encoder-neo1.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Inference function
def get_response(user_input, model, vectorizer, label_encoder):
    input_vector = vectorizer.transform([user_input]).toarray()
    predicted_label = int(round(model.predict(input_vector)[0]))  # Round to nearest response index
    return label_encoder.inverse_transform([predicted_label])[0]


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatai")
def chatai():
    return render_template("chatai.html")

@app.route("/get_response", methods=["POST"])
def chat_response():
    data = request.get_json()
    user_input = data.get("message", "")
    if user_input:
        response_text = get_response(user_input, model, vectorizer, label_encoder)
        return jsonify({"response": response_text})
    else:
        return jsonify({"response": "Please provide input."})

if __name__ == "__main__":
    app.run(debug=True)
