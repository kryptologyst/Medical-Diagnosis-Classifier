"""
Input validation utilities for medical diagnosis classifier
"""

from typing import Dict, List, Tuple, Any
import pandas as pd

class MedicalDataValidator:
    """Validates medical input data for the diagnosis classifier"""
    
    # Define valid ranges for each parameter
    PARAMETER_RANGES = {
        'Age': {'min': 18, 'max': 100, 'type': int},
        'Gender': {'min': 0, 'max': 1, 'type': int, 'values': [0, 1]},
        'BMI': {'min': 15.0, 'max': 50.0, 'type': float},
        'BloodPressure': {'min': 80, 'max': 200, 'type': int},
        'Glucose': {'min': 60, 'max': 300, 'type': int},
        'FamilyHistory': {'min': 0, 'max': 1, 'type': int, 'values': [0, 1]},
        'PhysicalActivity': {'min': 0, 'max': 2, 'type': int, 'values': [0, 1, 2]},
        'SmokingStatus': {'min': 0, 'max': 2, 'type': int, 'values': [0, 1, 2]}
    }
    
    REQUIRED_FIELDS = list(PARAMETER_RANGES.keys())
    
    @classmethod
    def validate_patient_data(cls, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate patient data for prediction
        
        Args:
            data: Dictionary containing patient data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for missing fields
        missing_fields = [field for field in cls.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate each field
        for field, value in data.items():
            if field not in cls.PARAMETER_RANGES:
                errors.append(f"Unknown field: {field}")
                continue
                
            field_errors = cls._validate_field(field, value)
            errors.extend(field_errors)
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_field(cls, field: str, value: Any) -> List[str]:
        """Validate a single field"""
        errors = []
        field_config = cls.PARAMETER_RANGES[field]
        
        # Check if value is None or empty
        if value is None or value == '':
            errors.append(f"{field} cannot be empty")
            return errors
        
        # Type validation
        try:
            if field_config['type'] == int:
                value = int(float(value))  # Handle string numbers
            elif field_config['type'] == float:
                value = float(value)
        except (ValueError, TypeError):
            errors.append(f"{field} must be a valid number")
            return errors
        
        # Range validation
        if value < field_config['min'] or value > field_config['max']:
            errors.append(f"{field} must be between {field_config['min']} and {field_config['max']}")
        
        # Specific value validation (for categorical fields)
        if 'values' in field_config and value not in field_config['values']:
            errors.append(f"{field} must be one of: {field_config['values']}")
        
        # Field-specific validations
        field_specific_errors = cls._validate_field_specific(field, value)
        errors.extend(field_specific_errors)
        
        return errors
    
    @classmethod
    def _validate_field_specific(cls, field: str, value: Any) -> List[str]:
        """Field-specific validation rules"""
        errors = []
        
        if field == 'BMI':
            if value < 16:
                errors.append("BMI below 16 indicates severe underweight - please consult a healthcare provider")
            elif value > 40:
                errors.append("BMI above 40 indicates severe obesity - please consult a healthcare provider")
        
        elif field == 'BloodPressure':
            if value > 180:
                errors.append("Blood pressure above 180 indicates hypertensive crisis - seek immediate medical attention")
            elif value < 90:
                errors.append("Blood pressure below 90 may indicate hypotension - consult a healthcare provider")
        
        elif field == 'Glucose':
            if value > 250:
                errors.append("Glucose level above 250 mg/dL is dangerously high - seek immediate medical attention")
            elif value < 70:
                errors.append("Glucose level below 70 mg/dL indicates hypoglycemia - consult a healthcare provider")
        
        elif field == 'Age':
            if value < 18:
                errors.append("This tool is designed for adults 18 years and older")
            elif value > 90:
                errors.append("For patients over 90, additional medical considerations may apply")
        
        return errors
    
    @classmethod
    def sanitize_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and convert data types"""
        sanitized = {}
        
        for field, value in data.items():
            if field in cls.PARAMETER_RANGES:
                field_config = cls.PARAMETER_RANGES[field]
                try:
                    if field_config['type'] == int:
                        sanitized[field] = int(float(value))
                    elif field_config['type'] == float:
                        sanitized[field] = float(value)
                    else:
                        sanitized[field] = value
                except (ValueError, TypeError):
                    sanitized[field] = value  # Keep original if conversion fails
            else:
                sanitized[field] = value
        
        return sanitized
    
    @classmethod
    def get_field_info(cls, field: str) -> Dict[str, Any]:
        """Get information about a specific field"""
        if field not in cls.PARAMETER_RANGES:
            return {}
        
        config = cls.PARAMETER_RANGES[field].copy()
        
        # Add human-readable descriptions
        descriptions = {
            'Age': 'Patient age in years (18-100)',
            'Gender': 'Biological sex (0=Female, 1=Male)',
            'BMI': 'Body Mass Index in kg/m² (15.0-50.0)',
            'BloodPressure': 'Systolic blood pressure in mmHg (80-200)',
            'Glucose': 'Fasting glucose level in mg/dL (60-300)',
            'FamilyHistory': 'Family history of diabetes (0=No, 1=Yes)',
            'PhysicalActivity': 'Physical activity level (0=Low, 1=Moderate, 2=High)',
            'SmokingStatus': 'Smoking status (0=Never, 1=Former, 2=Current)'
        }
        
        config['description'] = descriptions.get(field, '')
        return config
    
    @classmethod
    def validate_batch_data(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a batch of patient data (DataFrame)"""
        errors = []
        
        # Check required columns
        missing_cols = [col for col in cls.REQUIRED_FIELDS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return False, errors
        
        # Validate each row
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            is_valid, row_errors = cls.validate_patient_data(row_data)
            if not is_valid:
                errors.extend([f"Row {idx + 1}: {error}" for error in row_errors])
        
        return len(errors) == 0, errors

# Utility functions for common validations
def is_valid_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_string(text: str, max_length: int = 100) -> str:
    """Sanitize string input"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove potentially harmful characters
    import re
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    return sanitized[:max_length].strip()

def validate_file_upload(file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'csv', 'xlsx', 'json'}
    if '.' not in file.filename:
        return False, "File must have an extension"
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    # Check file size (limit to 10MB)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if size > 10 * 1024 * 1024:  # 10MB
        return False, "File size too large (max 10MB)"
    
    return True, "File is valid"
