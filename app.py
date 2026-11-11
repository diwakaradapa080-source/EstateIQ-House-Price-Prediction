from pathlib import Path
from datetime import datetime
import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_PATH = BASE_DIR / "house_prices.csv"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "parking_spaces"]
DEFAULTS = {"area_sqft": "1200", "bedrooms": "2", "bathrooms": "2", "age_years": "5", "location_score": "7", "parking_spaces": "1"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-demo-secret-change-before-deployment")

def ensure_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        from train_model import train_and_save
        train_and_save()
    return joblib.load(MODEL_PATH)

model = ensure_model()

def load_metadata():
    try:
        return json.loads(META_PATH.read_text())
    except Exception:
        return {"best_model": "Not available", "metrics": {}, "dataset_rows": 0, "train_rows": 0, "test_rows": 0}

def dataset_summary():
    data = pd.read_csv(DATA_PATH)
    return {
        "rows": len(data),
        "avg_price": float(data["price_inr"].mean()),
        "min_price": float(data["price_inr"].min()),
        "max_price": float(data["price_inr"].max()),
        "avg_area": float(data["area_sqft"].mean()),
        "median_price": float(data["price_inr"].median())
    }

def parse_features(form):
    values = {key: form.get(key, DEFAULTS[key]).strip() for key in DEFAULTS}
    try:
        numbers = {key: float(values[key]) for key in FEATURES}
    except (ValueError, TypeError):
        raise ValueError("Please enter valid numbers in every field.")
    if numbers["area_sqft"] <= 0 or numbers["area_sqft"] > 50000:
        raise ValueError("Area must be between 1 and 50,000 sq. ft.")
    for key in ("bedrooms", "bathrooms"):
        if not 1 <= numbers[key] <= 20 or not numbers[key].is_integer():
            raise ValueError(f"{key.replace('_', ' ').title()} must be a whole number from 1 to 20.")
    if not 0 <= numbers["age_years"] <= 200:
        raise ValueError("Property age must be between 0 and 200 years.")
    if not 1 <= numbers["location_score"] <= 10:
        raise ValueError("Location score must be between 1 and 10.")
    if not 0 <= numbers["parking_spaces"] <= 20 or not numbers["parking_spaces"].is_integer():
        raise ValueError("Parking spaces must be a whole number from 0 to 20.")
    return values, numbers

@app.route("/", methods=["GET", "POST"])
def home():
    values = DEFAULTS.copy()
    prediction = None
    error = None
    details = None
    if request.method == "POST":
        try:
            values, numbers = parse_features(request.form)
            row = pd.DataFrame([[numbers[k] for k in FEATURES]], columns=FEATURES)
            prediction = max(0, round(float(model.predict(row)[0])))
            details = {
                "price": prediction,
                "area": numbers["area_sqft"],
                "model": load_metadata().get("best_model", "Selected model"),
                "created": datetime.now().strftime("%d %b %Y, %I:%M %p")
            }
            history = session.get("history", [])
            history.insert(0, details)
            session["history"] = history[:5]
        except ValueError as exc:
            error = str(exc)
    meta = load_metadata()
    return render_template("index.html", values=values, prediction=prediction, error=error,
                           details=details, meta=meta, history=session.get("history", []))

@app.route("/dashboard")
def dashboard():
    meta = load_metadata()
    summary = dataset_summary()
    return render_template("dashboard.html", meta=meta, summary=summary)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        _, numbers = parse_features({k: str(payload.get(k, DEFAULTS[k])) for k in DEFAULTS})
        row = pd.DataFrame([[numbers[k] for k in FEATURES]], columns=FEATURES)
        price = max(0, round(float(model.predict(row)[0])))
        return jsonify({"success": True, "predicted_price_inr": price,
                        "model": load_metadata().get("best_model")})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
