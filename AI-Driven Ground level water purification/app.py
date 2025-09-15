from flask import Flask, request, jsonify
import joblib
import numpy as np


model = joblib.load("groundwater_best_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        data["pH"], data["TDS"], data["turbidity"], data["nitrate"],
        data["arsenic"], data["lead"], data["fluoride"], data["hardness"],
        data["temperature"], data["EC"], data["coliform"]
    ]
    features_array = np.array([features])

    pred = model.predict(features_array)[0]  # 0 = Unsafe, 1 = Safe

    result = "Safe" if pred == 1 else "Unsafe"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run()

