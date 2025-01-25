import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict
import os

class HealthcareModelTrainer:
    def __init__(self):
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Expanded symptom list with severity
        self.symptoms_meta = {
            'fever': {'severity_levels': ['mild', 'moderate', 'high']},
            'cough': {'severity_levels': ['dry', 'wet', 'severe']},
            'shortness_of_breath': {'severity_levels': ['mild', 'moderate', 'severe']},
            'fatigue': {'severity_levels': ['mild', 'moderate', 'severe']},
            'body_aches': {'severity_levels': ['mild', 'moderate', 'severe']},
            'sore_throat': {'severity_levels': ['mild', 'moderate', 'severe']},
            'runny_nose': {'severity_levels': ['mild', 'moderate', 'severe']},
            'congestion': {'severity_levels': ['mild', 'moderate', 'severe']},
            'headache': {'severity_levels': ['mild', 'moderate', 'severe']},
            'nausea': {'severity_levels': ['mild', 'moderate', 'severe']},
            'diarrhea': {'severity_levels': ['mild', 'moderate', 'severe']},
            'vomiting': {'severity_levels': ['mild', 'moderate', 'severe']},
            'loss_of_taste': {'severity_levels': ['partial', 'complete']},
            'loss_of_smell': {'severity_levels': ['partial', 'complete']},
            'chest_pain': {'severity_levels': ['mild', 'moderate', 'severe']},
            'abdominal_pain': {'severity_levels': ['mild', 'moderate', 'severe']}
        }
        
        # Define conditions with weighted symptom probabilities
        self.conditions = {
            'Common Cold': {
                'primary': {
                    'runny_nose': 0.9,
                    'congestion': 0.85,
                    'sore_throat': 0.8,
                    'cough': 0.7,
                    'fatigue': 0.6
                },
                'secondary': {
                    'fever': 0.3,
                    'headache': 0.4,
                    'body_aches': 0.3
                }
            },
            'Flu': {
                'primary': {
                    'fever': 0.9,
                    'body_aches': 0.85,
                    'fatigue': 0.9,
                    'cough': 0.7
                },
                'secondary': {
                    'sore_throat': 0.5,
                    'headache': 0.6,
                    'runny_nose': 0.4,
                    'nausea': 0.3
                }
            },
            'COVID-19': {
                'primary': {
                    'fever': 0.8,
                    'cough': 0.8,
                    'fatigue': 0.7,
                    'loss_of_taste': 0.6,
                    'loss_of_smell': 0.6
                },
                'secondary': {
                    'shortness_of_breath': 0.4,
                    'body_aches': 0.5,
                    'headache': 0.4,
                    'sore_throat': 0.3
                }
            },
            'Gastroenteritis': {
                'primary': {
                    'nausea': 0.9,
                    'vomiting': 0.8,
                    'diarrhea': 0.9,
                    'abdominal_pain': 0.8
                },
                'secondary': {
                    'fever': 0.4,
                    'fatigue': 0.6,
                    'headache': 0.3
                }
            },
            'Allergies': {
                'primary': {
                    'runny_nose': 0.9,
                    'congestion': 0.9,
                    'sore_throat': 0.6
                },
                'secondary': {
                    'cough': 0.4,
                    'headache': 0.3,
                    'fatigue': 0.3
                }
            }
        }
        
        self.standard_symptoms = list(self.symptoms_meta.keys())
    
    def generate_synthetic_diagnosis_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate more realistic synthetic diagnosis data"""
        np.random.seed(42)
        
        # Create empty dataframe
        df = pd.DataFrame(0, index=range(n_samples), columns=self.standard_symptoms + ['duration_days'])
        diagnoses = []
        
        for i in range(n_samples):
            # Random condition with equal probability
            condition = np.random.choice(list(self.conditions.keys()))
            diagnoses.append(condition)
            
            # Add primary symptoms with their probabilities
            for symptom, prob in self.conditions[condition]['primary'].items():
                if np.random.random() < prob:
                    # Add severity
                    severity = np.random.choice(self.symptoms_meta[symptom]['severity_levels'])
                    df.loc[i, symptom] = self.symptoms_meta[symptom]['severity_levels'].index(severity) + 1
            
            # Add secondary symptoms with their probabilities
            for symptom, prob in self.conditions[condition].get('secondary', {}).items():
                if np.random.random() < prob:
                    severity = np.random.choice(self.symptoms_meta[symptom]['severity_levels'])
                    df.loc[i, symptom] = self.symptoms_meta[symptom]['severity_levels'].index(severity) + 1
            
            # Add random duration (days)
            df.loc[i, 'duration_days'] = np.random.randint(1, 14)
            
            # Add some random noise (rare symptoms)
            other_symptoms = set(self.standard_symptoms) - set(self.conditions[condition]['primary'].keys()) - set(self.conditions[condition].get('secondary', {}).keys())
            for symptom in other_symptoms:
                if np.random.random() < 0.05:  # 5% chance
                    severity = np.random.choice(self.symptoms_meta[symptom]['severity_levels'])
                    df.loc[i, symptom] = self.symptoms_meta[symptom]['severity_levels'].index(severity) + 1
        
        df['diagnosis'] = diagnoses
        print("\n --- generate_synthetic_diagnosis_data\n", df.head())
        return df
    
    def train_diagnosis_model(self) -> Tuple[RandomForestClassifier, StandardScaler]:
        """Train the improved diagnosis model"""
        # Generate more diverse synthetic data
        df = self.generate_synthetic_diagnosis_data()
        
        # Prepare features and target
        X = df[self.standard_symptoms + ['duration_days']]
        y = df['diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with more trees and better parameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"Diagnosis Model Training Score: {train_score:.3f}")
        print(f"Diagnosis Model Test Score: {test_score:.3f}")
        
        return model, scaler
    
    def save_models(self, diagnosis_model: RandomForestClassifier, diagnosis_scaler: StandardScaler):
        """Save trained model and scaler"""
        joblib.dump(diagnosis_model, f'{self.models_dir}/diagnosis_model.joblib')
        joblib.dump(diagnosis_scaler, f'{self.models_dir}/diagnosis_scaler.joblib')
        
        print("Model and scaler saved successfully!")

def main():
    trainer = HealthcareModelTrainer()
    
    # Train diagnosis model
    print("Training improved diagnosis model...")
    diagnosis_model, diagnosis_scaler = trainer.train_diagnosis_model()

    # Save models
    trainer.save_models(diagnosis_model, diagnosis_scaler)
    
    # Example predictions with different symptom combinations
    print("\nTesting different symptom combinations:")
    
    test_cases = [
        {
            'name': "Cold-like symptoms",
            'symptoms': {
                'runny_nose': 2,
                'congestion': 2,
                'sore_throat': 1,
                'cough': 1,
                'duration_days': 3
            }
        },
        {
            'name': "Flu-like symptoms",
            'symptoms': {
                'fever': 2,
                'body_aches': 2,
                'fatigue': 2,
                'cough': 1,
                'duration_days': 5
            }
        },
        {
            'name': "Gastro symptoms",
            'symptoms': {
                'nausea': 2,
                'vomiting': 2,
                'diarrhea': 2,
                'abdominal_pain': 2,
                'duration_days': 2
            }
        }
    ]
    
    for test_case in test_cases:
        # Create feature vector
        example_symptoms = pd.DataFrame(0, index=[0], columns=trainer.standard_symptoms + ['duration_days'])
        for symptom, severity in test_case['symptoms'].items():
            example_symptoms.loc[0, symptom] = severity
        
        # Make prediction
        example_symptoms_scaled = diagnosis_scaler.transform(example_symptoms)
        diagnosis_pred = diagnosis_model.predict_proba(example_symptoms_scaled)
        
        print(f"\n{test_case['name']}:")
        predictions = list(zip(diagnosis_model.classes_, diagnosis_pred[0]))
        predictions.sort(key=lambda x: x[1], reverse=True)
        for condition, prob in predictions:
            if prob > 0.01:  # Only show significant probabilities
                print(f"{condition}: {prob:.3f}")

if __name__ == "__main__":
    main()