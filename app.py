# from flask import Flask, request, jsonify, render_template
# import joblib, os, datetime, pandas as pd
# # from model_train import train_model
#
# app = Flask(__name__)
#
# MODEL_PATH = "models/iris_model.pkl"
# LOG_PATH = "data/prediction_logs.csv"
#
# os.makedirs("models", exist_ok=True)
# os.makedirs("data", exist_ok=True)
#
# def load_model():
#     # if not os.path.exists(MODEL_PATH):
#     #     train_model()
#     return joblib.load(MODEL_PATH)
#
# @app.route("/")
# def home():
#     return render_template("index.html")
#
# @app.route("/dashboard")
# def dashboard():
#     if os.path.exists(LOG_PATH):
#         df = pd.read_csv(LOG_PATH)
#         df = df.tail(10)  # show last 10 predictions
#     else:
#         df = pd.DataFrame(columns=["timestamp", "features", "prediction"])
#     return render_template("dashboard.html", data=df.to_dict(orient="records"))
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json() or request.form
#     features = [
#         float(data.get("sepal_length")),
#         float(data.get("sepal_width")),
#         float(data.get("petal_length")),
#         float(data.get("petal_width")),
#     ]
#     model = load_model()
#     prediction = int(model.predict([features])[0])
#
#     # log
#     log = {
#         "timestamp": datetime.datetime.now().isoformat(),
#         "features": features,
#         "prediction": prediction,
#     }
#     df = pd.DataFrame([log])
#     df.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)
#
#     if request.is_json:
#         return jsonify({"prediction": prediction})
#     else:
#         return render_template("index.html", prediction=prediction)
#
# @app.route("/train", methods=["POST"])
# def retrain():
#     # acc = train_model()
#     return jsonify({"message": "Model retrained"})
#
# @app.route("/health")
# def health():
#     return jsonify({"status": "ok", "message": "Flask MLOps API running"})
#
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)

from flask import Flask, request, jsonify, render_template
import joblib
import os
import datetime
import pandas as pd
# from model_train import train_model  # Uncomment if retraining function is available

app = Flask(__name__)

MODEL_PATH = "models/iris_model.pkl"
LOG_PATH = "data/prediction_logs.csv"

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def load_model():
    """Load trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please train the model first.")
    return joblib.load(MODEL_PATH)

@app.route("/")
def home():
    """Render home page"""
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Show recent predictions"""
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = df.tail(10)
    else:
        df = pd.DataFrame(columns=["timestamp", "features", "prediction"])
    return render_template("dashboard.html", data=df.to_dict(orient="records"))

@app.route("/predict", methods=["POST"])
def predict():
    """Predict Iris class"""
    # Ensure content type is JSON if using API clients like Postman
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    try:
        features = [
            float(data.get("sepal_length")),
            float(data.get("sepal_width")),
            float(data.get("petal_length")),
            float(data.get("petal_width")),
        ]
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing feature values"}), 400

    model = load_model()
    prediction = model.predict([features])[0]

    # Log predictions
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "features": features,
        "prediction": prediction
    }
    pd.DataFrame([log_entry]).to_csv(
        LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False
    )

    # Return JSON if API, otherwise render template
    if request.is_json:
        return jsonify({"prediction": int(prediction)})
    else:
        return render_template("index.html", prediction=int(prediction))

@app.route("/train", methods=["POST"])
def retrain():
    """Retrain model (placeholder)"""
    # acc = train_model()
    return jsonify({"message": "Model retrained successfully"})

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Flask MLOps API running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
