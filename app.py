"""
Flask Web Application for Medical Diagnosis Classifier
Modern web interface for diabetes risk prediction
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
import pandas as pd
import numpy as np
from models.medical_classifier import MedicalDiagnosisClassifier
from utils.validation import MedicalDataValidator
import json
import logging
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'medical_diagnosis_secret_key_2024'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the classifier
classifier = MedicalDiagnosisClassifier()

# Global variable to track if model is loaded
model_loaded = False

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

def load_or_train_model():
    """Load existing model or train a new one"""
    global model_loaded
    
    try:
        # Try to load existing model
        classifier.load_model('models/medical_classifier.pkl')
        model_loaded = True
        print("Model loaded successfully")
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        # Load data and train model
        df = classifier.load_data('data/medical_dataset.csv')
        X, y = classifier.preprocess_data(df)
        classifier.train_model(X, y, model_type='logistic')
        classifier.save_model()
        model_loaded = True
        print("New model trained and saved")

@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model_loaded:
        logger.error("Prediction attempted with no model loaded")
        return jsonify({'error': 'Model not loaded. Please try again later.'}), 500
    
    try:
        # Get form data
        raw_data = {
            'Age': request.form.get('age'),
            'Gender': request.form.get('gender'),
            'BMI': request.form.get('bmi'),
            'BloodPressure': request.form.get('blood_pressure'),
            'Glucose': request.form.get('glucose'),
            'FamilyHistory': request.form.get('family_history'),
            'PhysicalActivity': request.form.get('physical_activity'),
            'SmokingStatus': request.form.get('smoking_status')
        }
        
        # Validate input data
        is_valid, validation_errors = MedicalDataValidator.validate_patient_data(raw_data)
        if not is_valid:
            logger.warning(f"Validation failed: {validation_errors}")
            return jsonify({
                'error': 'Invalid input data',
                'details': validation_errors
            }), 400
        
        # Sanitize data
        patient_data = MedicalDataValidator.sanitize_data(raw_data)
        
        # Make prediction
        result = classifier.predict_single_patient(patient_data)
        
        # Add patient data to result for display
        result['patient_data'] = patient_data
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Successful prediction: {result['risk_level']}")
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not model_loaded:
        logger.error("API prediction attempted with no model loaded")
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate input data
        is_valid, validation_errors = MedicalDataValidator.validate_patient_data(data)
        if not is_valid:
            logger.warning(f"API validation failed: {validation_errors}")
            return jsonify({
                'error': 'Invalid input data',
                'details': validation_errors
            }), 400
        
        # Sanitize data
        patient_data = MedicalDataValidator.sanitize_data(data)
        
        # Make prediction
        result = classifier.predict_single_patient(patient_data)
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Successful API prediction: {result['risk_level']}")
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"API value error: {e}")
        return jsonify({'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"API unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/model_info')
def model_info():
    """Display model information"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(classifier.training_history)

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/api_docs')
def api_docs():
    """API documentation page"""
    return render_template('api_docs.html')

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
