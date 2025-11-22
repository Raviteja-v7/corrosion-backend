"""
Corrosion Prediction Backend - Flask API with ML Model
Install required packages:
pip install flask flask-cors numpy pandas scikit-learn joblib
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
CORS(app)  # Enable CORS for React frontend

# Global variables for model and preprocessors
model = None
scaler = None
material_encoder = None
env_encoder = None

def generate_training_data(n_samples=1000):
    """Generate synthetic training data for corrosion prediction"""
    np.random.seed(42)
    
    data = []
    materials = ['steel', 'aluminum', 'copper', 'stainless-steel', 'galvanized-steel']
    environments = ['urban', 'marine', 'industrial', 'rural']
    
    for _ in range(n_samples):
        # Generate random input parameters
        temperature = np.random.uniform(0, 50)
        humidity = np.random.uniform(30, 100)
        pH = np.random.uniform(2, 12)
        chloride = np.random.uniform(0, 5)
        material = np.random.choice(materials)
        exposure_time = np.random.uniform(1, 60)
        coating = np.random.uniform(0, 200)
        environment = np.random.choice(environments)
        
        # Calculate corrosion rate with realistic physics-based formula
        # Base corrosion rate
        base_rate = 0.05
        
        # Temperature effect (Arrhenius-like behavior)
        temp_factor = 1 + (temperature - 25) * 0.03
        
        # Humidity effect (exponential above 60%)
        if humidity > 60:
            humidity_factor = 1 + (humidity - 60) * 0.02
        else:
            humidity_factor = 1
        
        # pH effect (corrosion increases away from neutral)
        ph_factor = 1 + abs(7 - pH) * 0.15
        
        # Chloride effect (aggressive)
        chloride_factor = 1 + chloride * 0.4
        
        # Material resistance factors
        material_factors = {
            'steel': 1.0,
            'aluminum': 0.6,
            'copper': 0.7,
            'stainless-steel': 0.25,
            'galvanized-steel': 0.4
        }
        
        # Environmental multipliers
        env_factors = {
            'urban': 1.0,
            'marine': 1.8,
            'industrial': 2.0,
            'rural': 0.6
        }
        
        # Coating protection (exponential decay)
        coating_protection = np.exp(-coating / 100)
        coating_factor = 0.3 + 0.7 * coating_protection
        
        # Calculate corrosion rate (mm/year)
        corrosion_rate = (base_rate * temp_factor * humidity_factor * 
                         ph_factor * chloride_factor * coating_factor *
                         material_factors[material] * env_factors[environment])
        
        # Add some noise for realism
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

def train_model():
    """Train the ML model for corrosion prediction"""
    global model, scaler, material_encoder, env_encoder
    
    print("Generating training data...")
    df = generate_training_data(n_samples=2000)
    
    # Encode categorical variables
    material_encoder = LabelEncoder()
    env_encoder = LabelEncoder()
    
    df['material_encoded'] = material_encoder.fit_transform(df['material_type'])
    df['environment_encoded'] = env_encoder.fit_transform(df['environmental_condition'])
    
    # Prepare features and target
    feature_columns = ['temperature', 'humidity', 'pH', 'chloride_concentration',
                      'exposure_time', 'coating_thickness', 'material_encoded', 
                      'environment_encoded']
    
    X = df[feature_columns]
    y = df['corrosion_rate']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble model (Random Forest + Gradient Boosting)
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    # Use Random Forest as primary model (typically performs better)
    model = rf_model
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # Save model and preprocessors
    joblib.dump(model, 'corrosion_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(material_encoder, 'material_encoder.pkl')
    joblib.dump(env_encoder, 'env_encoder.pkl')
    
    print("\nModel trained and saved successfully!")
    return model, scaler, material_encoder, env_encoder

def load_model():
    """Load pre-trained model and preprocessors"""
    global model, scaler, material_encoder, env_encoder
    
    if os.path.exists('corrosion_model.pkl'):
        print("Loading existing model...")
        model = joblib.load('corrosion_model.pkl')
        scaler = joblib.load('scaler.pkl')
        material_encoder = joblib.load('material_encoder.pkl')
        env_encoder = joblib.load('env_encoder.pkl')
        print("Model loaded successfully!")
    else:
        print("No existing model found. Training new model...")
        train_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict corrosion rate based on input parameters"""
    try:
        data = request.json
        
        # Extract parameters
        temperature = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 60))
        pH = float(data.get('pH', 7))
        chloride = float(data.get('chlorideConcentration', 0.5))
        material = data.get('materialType', 'steel')
        exposure_time = float(data.get('exposureTime', 12))
        coating = float(data.get('coatingThickness', 50))
        environment = data.get('environmentalCondition', 'urban')
        
        # Encode categorical variables
        material_encoded = material_encoder.transform([material])[0]
        env_encoded = env_encoder.transform([environment])[0]
        
        # Prepare features
        features = np.array([[
            temperature, humidity, pH, chloride,
            exposure_time, coating, material_encoded, env_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict corrosion rate
        corrosion_rate = model.predict(features_scaled)[0]
        
        # Generate forecast data
        forecast_data = []
        for month in range(int(exposure_time) + 1):
            monthly_corrosion = corrosion_rate * month / 12
            forecast_data.append({
                'month': month,
                'corrosion': round(monthly_corrosion, 3),
                'threshold': 2.0
            })
        
        # Calculate total corrosion
        total_corrosion = corrosion_rate * exposure_time / 12
        
        # Risk assessment
        if total_corrosion > 2.0:
            risk_level = 'High'
            risk_color = 'text-red-600'
        elif total_corrosion > 1.0:
            risk_level = 'Medium'
            risk_color = 'text-yellow-600'
        else:
            risk_level = 'Low'
            risk_color = 'text-green-600'
        
        # Calculate time to threshold
        if corrosion_rate > 0:
            time_to_threshold = (2.0 / corrosion_rate) * 12  # in months
        else:
            time_to_threshold = 999
        
        # Feature importance (simplified)
        contributing_factors = [
            {'name': 'Temperature', 'impact': abs(temperature - 25) * 2, 'value': f"{temperature}°C"},
            {'name': 'Humidity', 'impact': abs(humidity - 50) * 1.5, 'value': f"{humidity}%"},
            {'name': 'pH Level', 'impact': abs(pH - 7) * 10, 'value': str(pH)},
            {'name': 'Chloride', 'impact': chloride * 15, 'value': f"{chloride} mg/L"},
            {'name': 'Coating', 'impact': (100 - coating) * 0.5, 'value': f"{coating}μm"}
        ]
        contributing_factors.sort(key=lambda x: x['impact'], reverse=True)
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': {
                'corrosionRate': round(corrosion_rate, 3),
                'totalCorrosion': round(total_corrosion, 2),
                'riskLevel': risk_level,
                'riskColor': risk_color,
                'timeToThreshold': round(time_to_threshold, 1),
                'forecastData': forecast_data,
                'contributingFactors': contributing_factors,
                'confidence': 0.85  # You can calculate this from model uncertainty
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain the model with new data"""
    try:
        train_model()
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the trained model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404
    
    return jsonify({
        'model_type': type(model).__name__,
        'features': ['temperature', 'humidity', 'pH', 'chloride_concentration',
                    'exposure_time', 'coating_thickness', 'material_type', 
                    'environmental_condition'],
        'materials': material_encoder.classes_.tolist(),
        'environments': env_encoder.classes_.tolist()
    })

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    
    # Run Flask app
    print("\n" + "="*50)
    print("Corrosion Prediction API Server")
    print("="*50)
    print("API Endpoints:")
    print("  - POST /api/predict - Make corrosion predictions")
    print("  - GET  /api/health - Health check")
    print("  - GET  /api/model-info - Model information")
    print("  - POST /api/retrain - Retrain the model")
    print("="*50 + "\n")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))