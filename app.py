from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from React (localhost:5173)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("expenses", [])  # Expecting [{amount: 50}, {amount: 60}]
    if len(data) < 2:
        return jsonify({"error": "Need at least 2 weeks of data"}), 400

    weeks = np.array(range(1, len(data) + 1)).reshape(-1, 1)  # [[1], [2], ...]
    amounts = np.array([exp["amount"] for exp in data])       # [50, 60, ...]
    model = LinearRegression()
    model.fit(weeks, amounts)
    next_week = model.predict([[len(data) + 1]])
    return jsonify({"prediction": round(next_week[0], 2)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)