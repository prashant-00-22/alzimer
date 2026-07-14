from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load model & scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    print("Model files not found. Run train_model.py first.")
    model, scaler = None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    gender_male = 1 if data['genderMale'] else 0
    race_white = 1 if data['raceWhite'] else 0
    education = float(data['education'])
    apoe4 = int(data['apoe4'])
    
    input_data = pd.DataFrame({
        'PTGENDER_Male': [gender_male],
        'PTEDUCAT': [education],
        'PTRACCAT_White': [race_white],
        'APOE4': [apoe4]
    }, columns=['PTGENDER_Male', 'PTEDUCAT', 'PTRACCAT_White', 'APOE4'])
    
    input_data[['PTEDUCAT']] = scaler.transform(input_data[['PTEDUCAT']])
    
    prob = model.predict_proba(input_data)[0,1]
    risk = "High Risk (AD)" if prob > 0.1 else "Low Risk"
    
    features = ['Gender (Male)', 'Education (Years)', 'Race (White)', 'APOE4']
    importance = model.feature_importances_.tolist()
    
    return jsonify({
        'probability': round(prob, 3),
        'riskLevel': risk,
        'featureImportance': importance,
        'features': features
    })

if __name__ == '__main__':
    print("Starting Alzheimer's Predictor on http://localhost:5000")
    print("Ctrl+C to stop.")
    app.run(debug=True, host='0.0.0.0', port=5000)
