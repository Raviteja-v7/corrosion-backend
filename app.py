"""
Corrosion Prediction Backend - Flask API with ML Model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
material_encoder = None
env_encoder = None

# -----------------------------
# Training Data Generator
# -----------------------------
def generate_training_data(n_samples=2000):
    np.random.seed(42)

    data = []
    materials = ['steel', 'aluminum', 'copper', 'stainless-steel', 'galvanized-steel']
    environments = ['urban', 'marine', 'industrial', 'rural']

    for _ in range(n_samples):
        temperature = np.random.uniform(0, 50)
        humidity = np.random.uniform(30, 100)
        pH = np.random.uniform(2, 12)
        chloride = np.random.uniform(0, 5)
        material = np.random.choice(materials)
        exposure_time = np.random.uniform(1, 60)
        coating = np.random.uniform(0, 200)
        environment = np.random.choice(environments)

        base_rate = 0.05
        temp_factor = 1 + (temperature - 25) * 0.03
        humidity_factor = 1 + max(0, humidity - 60) * 0.02
        ph_factor = 1 + abs(7 - pH) * 0.15
        chloride_factor = 1 + chloride * 0.4

        material_factors = {
            'steel': 1.0, 'aluminum': 0.6, 'copper': 0.7,
            'stainless-steel': 0.25, 'galvanized-steel': 0.4
        }
        env_factors = {
            'urban': 1.0, 'marine': 1.8, 'industrial': 2.0, 'rural': 0.6
        }

        coating_factor = 0.3 + 0.7 * np.exp(-coating / 100)

        corrosion_rate = (base_rate * temp_factor * humidity_factor *
                          ph_factor * chloride_factor * coating_factor *
                          material_factors[material] * env_factors[environment])

        corrosion_rate *= np.random.uniform(0.85, 1.15)

        data.append({
            'temperature': temperature,
            'humidity': humidity,
            'pH': pH,
            'chloride_concentration': chloride,
            'material_type': material,
            'exposure_time': exposure_time,
            'coating_thickness': coating,
            'environmental_condition': environment,
            'corrosion_rate': corrosion_rate
        })

    return pd.DataFrame(data)

# -----------------------------
# Model Training
# -----------------------------
def train_model():
    global model, scaler, material_encoder, env_encoder

    print("Generating training data...")
    df = generate_training_data()

    material_encoder = LabelEncoder()
    env_encoder = LabelEncoder()

    df['material_encoded'] = material_encoder.fit_transform(df['material_type'])
    df['environment_encoded'] = env_encoder.fit_transform(df['environmental_condition'])

    features = ['temperature', 'humidity', 'pH', 'chloride_concentration',
                'exposure_time', 'coating_thickness',
                'material_encoded', 'environment_encoded']

    X = df[features]
    y = df['corrosion_rate']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Model trained. R²:", r2_score(y_test, y_pred))

    joblib.dump(model, "corrosion_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(material_encoder, "material_encoder.pkl")
    joblib.dump(env_encoder, "env_encoder.pkl")

    print("Saved model files.")

# -----------------------------
# Model Loader
# -----------------------------
def load_model():
    global model, scaler, material_encoder, env_encoder

    if os.path.exists("corrosion_model.pkl"):
        print("Loading model files...")
        model = joblib.load("corrosion_model.pkl")
        scaler = joblib.load("scaler.pkl")
        material_encoder = joblib.load("material_encoder.pkl")
        env_encoder = joblib.load("env_encoder.pkl")
        print("Model loaded.")
    else:
        print("No model found → Training new model...")
        train_model()

# ---------------------------------------------------
# IMPORTANT: Load the model at import (Gunicorn needs this)
# ---------------------------------------------------
load_model()

# -----------------------------
# API ROUTES
# -----------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        global model, scaler, material_encoder, env_encoder

        data = request.get_json()
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        pH = float(data["pH"])
        chloride = float(data["chlorideConcentration"])
        material = data["materialType"]
        exposure_time = float(data["exposureTime"])
        coating = float(data["coatingThickness"])
        environment = data["environmentalCondition"]

        material_encoded = material_encoder.transform([material])[0]
        env_encoded = env_encoder.transform([environment])[0]

        features = scaler.transform([[
            temperature, humidity, pH, chloride,
            exposure_time, coating,
            material_encoded, env_encoded
        ]])

        corrosion_rate = model.predict(features)[0]
        total_corrosion = corrosion_rate * exposure_time / 12

        # Risk assessment
        if total_corrosion > 2:
            risk_level = "High"
            risk_color = "text-red-600"
        elif total_corrosion > 1:
            risk_level = "Medium"
            risk_color = "text-yellow-600"
        else:
            risk_level = "Low"
            risk_color = "text-green-600"

        # Forecast curve
        forecast = [{
            "month": m,
            "corrosion": round(corrosion_rate * m / 12, 3),
            "threshold": 2.0
        } for m in range(int(exposure_time) + 1)]

        return jsonify({
            "success": True,
            "prediction": {
                "corrosionRate": round(corrosion_rate, 3),
                "totalCorrosion": round(total_corrosion, 2),
                "riskLevel": risk_level,
                "riskColor": risk_color,
                "timeToThreshold": round((2.0 / corrosion_rate) * 12, 1),
                "forecastData": forecast,
                "contributingFactors": [],
                "confidence": 0.85
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# -----------------------------
# Local Dev Server (ignored on Render)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)