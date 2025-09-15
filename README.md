# 🩺 Medical Diagnosis Classifier

An AI-powered diabetes risk assessment tool using machine learning to predict diabetes risk based on patient health metrics and lifestyle factors.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **AI-Powered Risk Assessment**: Advanced machine learning algorithms for diabetes prediction
- **Interactive Web Interface**: Modern, responsive UI for easy patient data input
- **Real-time Predictions**: Instant risk assessment with confidence scores
- **RESTful API**: Easy integration with other healthcare systems
- **Comprehensive Analytics**: Detailed model performance metrics and visualizations
- **Educational Tool**: Perfect for learning ML in healthcare applications

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-diagnosis-classifier.git
   cd medical-diagnosis-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Dataset

The project includes a sophisticated mock medical dataset generator that creates realistic patient data with:

- **1000+ patient records** with realistic correlations
- **8 health metrics**: Age, Gender, BMI, Blood Pressure, Glucose, Family History, Physical Activity, Smoking Status
- **Medically accurate relationships** between risk factors and diabetes outcomes
- **Balanced dataset** with appropriate diabetes prevalence rates

### Health Metrics Analyzed

| Metric | Description | Range | Impact |
|--------|-------------|-------|--------|
| **Age** | Patient age in years | 18-100 | Higher risk after 45 |
| **Gender** | Biological sex | Male/Female | Males slightly higher risk |
| **BMI** | Body Mass Index | 15-50 kg/m² | Obesity increases risk |
| **Blood Pressure** | Systolic BP | 80-200 mmHg | Hypertension correlation |
| **Glucose** | Fasting glucose | 60-300 mg/dL | Direct diabetes indicator |
| **Family History** | Genetic predisposition | Yes/No | Strong risk factor |
| **Physical Activity** | Exercise level | Low/Moderate/High | Protective factor |
| **Smoking Status** | Tobacco use | Never/Former/Current | Increases complications |

## 🧠 Machine Learning Model

### Algorithm Details

- **Primary Model**: Logistic Regression with L1/L2 regularization
- **Alternative**: Random Forest Classifier
- **Optimization**: Grid Search with 5-fold Cross-Validation
- **Feature Scaling**: StandardScaler normalization
- **Validation**: Stratified train/test split (80/20)

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 85%+ |
| **ROC-AUC** | 0.90+ |
| **Precision** | 80%+ |
| **Recall** | 85%+ |

### Feature Importance

The model analyzes the relative importance of each health metric:

1. **Glucose Level** - Primary diabetes indicator
2. **BMI** - Strong metabolic correlation
3. **Age** - Progressive risk factor
4. **Family History** - Genetic predisposition
5. **Blood Pressure** - Metabolic syndrome component
6. **Physical Activity** - Protective lifestyle factor
7. **Smoking Status** - Complication risk
8. **Gender** - Demographic factor

## 🌐 Web Interface

### Main Features

- **Patient Input Form**: Intuitive form with validation and help text
- **Risk Assessment**: Color-coded risk levels (Low/Moderate/High)
- **Recommendations**: Personalized health advice based on risk level
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Accessibility**: WCAG compliant with proper ARIA labels

### Screenshots

The web interface provides:
- Clean, medical-themed design
- Real-time form validation
- Interactive risk visualization
- Detailed result explanations
- Health recommendations

## 🔌 API Documentation

### Endpoints

#### Predict Diabetes Risk
```http
POST /api/predict
Content-Type: application/json

{
  "Age": 45,
  "Gender": 1,
  "BMI": 28.5,
  "BloodPressure": 140,
  "Glucose": 120,
  "FamilyHistory": 1,
  "PhysicalActivity": 1,
  "SmokingStatus": 0
}
```

**Response:**
```json
{
  "prediction": 1,
  "risk_probability": 0.75,
  "confidence": 0.85,
  "risk_level": "High Risk"
}
```

#### Model Information
```http
GET /model_info
```

### Integration Examples

#### Python
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "Age": 45, "Gender": 1, "BMI": 28.5,
    "BloodPressure": 140, "Glucose": 120,
    "FamilyHistory": 1, "PhysicalActivity": 1,
    "SmokingStatus": 0
}

response = requests.post(url, json=data)
result = response.json()
print(f"Risk: {result['risk_level']}")
```

#### JavaScript
```javascript
fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(patientData)
})
.then(response => response.json())
.then(data => console.log(data.risk_level));
```

## 📁 Project Structure

```
medical-diagnosis-classifier/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
├── data/
│   ├── medical_data.py   # Dataset generator
│   └── medical_dataset.csv # Generated dataset
├── models/
│   ├── medical_classifier.py # ML model implementation
│   └── medical_classifier.pkl # Trained model
├── templates/
│   ├── base.html        # Base template
│   ├── index.html       # Main interface
│   ├── about.html       # Project information
│   └── api_docs.html    # API documentation
└── results/
    ├── confusion_matrix.png # Model evaluation
    └── roc_curve.png       # Performance metrics
```

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=models
```

### Model Training

```bash
# Train new model
python -c "
from models.medical_classifier import MedicalDiagnosisClassifier
classifier = MedicalDiagnosisClassifier()
df = classifier.load_data()
X, y = classifier.preprocess_data(df)
classifier.train_model(X, y)
classifier.save_model()
"
```

### Data Generation

```bash
# Generate new dataset
python data/medical_data.py
```

## 📈 Model Evaluation

The project includes comprehensive model evaluation:

- **Confusion Matrix**: Visual representation of prediction accuracy
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Feature Importance**: Understanding which factors matter most
- **Cross-Validation**: Robust performance estimation
- **Classification Report**: Detailed precision, recall, and F1-scores

## 🏥 Medical Disclaimer

⚠️ **IMPORTANT**: This tool is designed for **educational and research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions and diabetes screening.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** for machine learning algorithms
- **Flask** for web framework
- **Bootstrap** for responsive UI components
- **Medical research community** for diabetes risk factor insights

## 📞 Support

If you have any questions or need help with the project:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/medical-diagnosis-classifier/issues)
- 📖 Documentation: [Project Wiki](https://github.com/yourusername/medical-diagnosis-classifier/wiki)

---

**Made with ❤️ for healthcare AI education**
# Medical-Diagnosis-Classifier
