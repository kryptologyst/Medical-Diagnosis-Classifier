"""
Medical Diagnosis Classifier
Advanced implementation with proper data handling, model evaluation, and prediction capabilities
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MedicalDiagnosisClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_type = None
        self.training_history = {}
        
    def load_data(self, data_path=None):
        """Load medical data from CSV or generate if not exists"""
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"Loaded data from {data_path}")
        else:
            # Generate data if file doesn't exist
            from data.medical_data import MedicalDataGenerator
            generator = MedicalDataGenerator()
            df = generator.generate_patient_data(1000)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/medical_dataset.csv', index=False)
            print("Generated new medical dataset")
            
        return df
    
    def preprocess_data(self, df):
        """Preprocess the medical data for training"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Remove PatientID as it's not a feature
        if 'PatientID' in data.columns:
            data = data.drop('PatientID', axis=1)
        
        # Separate features and target
        X = data.drop('DiabetesRisk', axis=1)
        y = data['DiabetesRisk']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self, X, y, model_type='logistic', test_size=0.2, random_state=42):
        """Train the medical diagnosis model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose and train model
        if model_type == 'logistic':
            # Grid search for best parameters
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            base_model = LogisticRegression(random_state=random_state, max_iter=1000)
            
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(random_state=random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        self.model_type = model_type
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store training history
        self.training_history = {
            'model_type': model_type,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_accuracy': self.model.score(X_test_scaled, y_test),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names
        }
        
        # Generate evaluation report
        self.evaluate_model(X_test_scaled, y_test, y_pred, y_prob)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, y_pred, y_prob):
        """Comprehensive model evaluation"""
        print("="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['No Diabetes', 'Diabetes']))
        
        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            print("\nFeature Importance:")
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            print(importance_df.to_string(index=False))
        
        elif hasattr(self.model, 'coef_'):
            print("\nFeature Coefficients:")
            coef_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
            print(coef_df.to_string(index=False))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot ROC curve
        self.plot_roc_curve(y_test, y_prob)
        
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - Medical Diagnosis Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_patient(self, patient_data):
        """Predict diabetes risk for a single patient"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in patient_df.columns:
                raise ValueError(f"Missing feature: {feature}")
        
        # Select and order features correctly
        patient_features = patient_df[self.feature_names]
        
        # Scale features
        patient_scaled = self.scaler.transform(patient_features)
        
        # Predict
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'risk_probability': float(probability[1]),
            'confidence': float(max(probability)),
            'risk_level': self.get_risk_level(probability[1])
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def save_model(self, filepath='models/medical_classifier.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/medical_classifier.pkl'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.training_history = model_data['training_history']
        
        print(f"Model loaded from {filepath}")

# Example usage and training
if __name__ == "__main__":
    # Initialize classifier
    classifier = MedicalDiagnosisClassifier()
    
    # Load data
    df = classifier.load_data('data/medical_dataset.csv')
    
    # Preprocess data
    X, y = classifier.preprocess_data(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train model
    print("\nTraining Logistic Regression model...")
    classifier.train_model(X, y, model_type='logistic')
    
    # Save model
    classifier.save_model()
    
    # Example prediction
    sample_patient = {
        'Age': 45,
        'Gender': 1,  # Male
        'BMI': 28.5,
        'BloodPressure': 140,
        'Glucose': 120,
        'FamilyHistory': 1,  # Yes
        'PhysicalActivity': 1,  # Moderate
        'SmokingStatus': 0  # Never
    }
    
    result = classifier.predict_single_patient(sample_patient)
    print(f"\nSample Prediction:")
    print(f"Patient Data: {sample_patient}")
    print(f"Diabetes Risk: {result['risk_level']}")
    print(f"Probability: {result['risk_probability']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
