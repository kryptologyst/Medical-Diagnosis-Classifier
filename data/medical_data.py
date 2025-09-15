"""
Mock Medical Database Generator
Generates realistic medical data for diabetes diagnosis classification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class MedicalDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_patient_data(self, n_patients=1000):
        """Generate realistic patient data for diabetes diagnosis"""
        
        # Age distribution (20-80 years, higher diabetes risk with age)
        ages = np.random.normal(50, 15, n_patients)
        ages = np.clip(ages, 20, 80).astype(int)
        
        # Gender (0: Female, 1: Male)
        genders = np.random.choice([0, 1], n_patients, p=[0.52, 0.48])
        
        # BMI (correlated with diabetes risk)
        # Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30+
        bmis = []
        for age in ages:
            if age < 30:
                bmi = np.random.normal(24, 4)
            elif age < 50:
                bmi = np.random.normal(27, 5)
            else:
                bmi = np.random.normal(29, 6)
            bmis.append(max(16, min(50, bmi)))
        
        # Blood Pressure (Systolic)
        blood_pressures = []
        for age, bmi in zip(ages, bmis):
            base_bp = 90 + (age - 20) * 0.5 + (bmi - 25) * 0.8
            bp = np.random.normal(base_bp, 15)
            blood_pressures.append(max(80, min(200, bp)))
        
        # Glucose levels (fasting glucose)
        glucose_levels = []
        diabetes_status = []
        
        for i, (age, bmi, bp) in enumerate(zip(ages, bmis, blood_pressures)):
            # Calculate diabetes probability based on risk factors
            diabetes_prob = 0.05  # Base probability
            
            # Age factor
            if age > 45:
                diabetes_prob += (age - 45) * 0.01
            
            # BMI factor
            if bmi > 25:
                diabetes_prob += (bmi - 25) * 0.02
            
            # Blood pressure factor
            if bp > 130:
                diabetes_prob += (bp - 130) * 0.002
            
            # Gender factor (males slightly higher risk)
            if genders[i] == 1:
                diabetes_prob += 0.02
            
            # Cap probability
            diabetes_prob = min(0.8, diabetes_prob)
            
            # Determine diabetes status
            has_diabetes = np.random.random() < diabetes_prob
            diabetes_status.append(1 if has_diabetes else 0)
            
            # Generate glucose level based on diabetes status
            if has_diabetes:
                glucose = np.random.normal(140, 25)  # Diabetic range
            else:
                glucose = np.random.normal(95, 15)   # Normal range
            
            glucose_levels.append(max(60, min(300, glucose)))
        
        # Family history (genetic factor)
        family_history = np.random.choice([0, 1], n_patients, p=[0.7, 0.3])
        
        # Physical activity level (0: Low, 1: Moderate, 2: High)
        activity_levels = np.random.choice([0, 1, 2], n_patients, p=[0.4, 0.4, 0.2])
        
        # Smoking status (0: Never, 1: Former, 2: Current)
        smoking_status = np.random.choice([0, 1, 2], n_patients, p=[0.6, 0.25, 0.15])
        
        # Create DataFrame
        data = {
            'PatientID': [f'P{i:04d}' for i in range(1, n_patients + 1)],
            'Age': ages,
            'Gender': genders,  # 0: Female, 1: Male
            'BMI': np.round(bmis, 1),
            'BloodPressure': np.round(blood_pressures, 0).astype(int),
            'Glucose': np.round(glucose_levels, 0).astype(int),
            'FamilyHistory': family_history,  # 0: No, 1: Yes
            'PhysicalActivity': activity_levels,  # 0: Low, 1: Moderate, 2: High
            'SmokingStatus': smoking_status,  # 0: Never, 1: Former, 2: Current
            'DiabetesRisk': diabetes_status  # 0: No diabetes, 1: Diabetes
        }
        
        return pd.DataFrame(data)
    
    def save_data(self, df, filename='medical_dataset.csv'):
        """Save the generated data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Medical dataset saved to {filename}")
        return filename
    
    def get_data_summary(self, df):
        """Get summary statistics of the generated data"""
        summary = {
            'total_patients': len(df),
            'diabetes_cases': df['DiabetesRisk'].sum(),
            'diabetes_rate': df['DiabetesRisk'].mean() * 100,
            'avg_age': df['Age'].mean(),
            'avg_bmi': df['BMI'].mean(),
            'avg_glucose': df['Glucose'].mean(),
            'gender_distribution': df['Gender'].value_counts().to_dict()
        }
        return summary

# Generate the dataset
if __name__ == "__main__":
    generator = MedicalDataGenerator()
    medical_df = generator.generate_patient_data(1000)
    
    # Save to CSV
    generator.save_data(medical_df, 'data/medical_dataset.csv')
    
    # Print summary
    summary = generator.get_data_summary(medical_df)
    print("\nDataset Summary:")
    print(f"Total Patients: {summary['total_patients']}")
    print(f"Diabetes Cases: {summary['diabetes_cases']}")
    print(f"Diabetes Rate: {summary['diabetes_rate']:.1f}%")
    print(f"Average Age: {summary['avg_age']:.1f}")
    print(f"Average BMI: {summary['avg_bmi']:.1f}")
    print(f"Average Glucose: {summary['avg_glucose']:.1f}")
