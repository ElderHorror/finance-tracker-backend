from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("expenses", [])
    if len(data) < 2:
        return jsonify({"error": "Need at least 2 weeks of data"}), 400
    weeks = np.array(range(1, len(data) + 1)).reshape(-1, 1)
    amounts = np.array([exp["amount"] for exp in data])
    model = LinearRegression()
    model.fit(weeks, amounts)
    next_week = model.predict([[len(data) + 1]])[0]
    avg = np.mean(amounts)  # Fallback average
    prediction = min(max(next_week, 0), avg * 2)  # Cap at 2x average, floor at 0
    return jsonify({"prediction": round(prediction, 2)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)