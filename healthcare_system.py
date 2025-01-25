from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import joblib


from patient_db_operations import PatientDatabase
from multi_agent import HealthcareMultiAgentSystem


class HealthcareSystem:
    def __init__(self):
        self.patient_db = PatientDatabase()
        self.healthcare_agent = HealthcareMultiAgentSystem()

        # standard symptoms
        self.standard_symptoms = self._load_standard_symptoms()
        
        # Load ML model
        self.model_version = 'v1.0.0'
        self.model = joblib.load('models/diagnosis_model.joblib')
        self.scaler = joblib.load('models/diagnosis_scaler.joblib')
        
        
    def _load_standard_symptoms(self) -> List[str]:
        # Load from the original trainer class
        return [
            'fever', 'cough', 'shortness_of_breath', 'fatigue', 'body_aches',
            'sore_throat', 'runny_nose', 'congestion', 'headache', 'nausea',
            'diarrhea', 'vomiting', 'loss_of_taste', 'loss_of_smell',
            'chest_pain', 'abdominal_pain'
        ]

    def analyze_symptom_trends(self, historical_data: List[Dict], current_features: Dict) -> Dict:
        """Analyze trends in symptoms over time, incorporating current symptoms"""
        symptom_trends = {}

        # Prepare historical symptom data
        for record in historical_data:
            symptoms = record['symptoms']
            date = record['created_at']
            
            for symptom, severity in symptoms.items():
                if symptom not in symptom_trends:
                    symptom_trends[symptom] = []
                symptom_trends[symptom].append((date, severity))

        # Calculate trend analysis
        trend_analysis = {}
        for symptom, values in symptom_trends.items():
            # Sort values by date to ensure chronological order
            values.sort(key=lambda x: x[0])  # Assuming x[0] is already a datetime object

            # Get the most recent historical severity if available
            previous_severity = values[-1][1] if values else None

            # Get current severity from current features
            current_severity = current_features.get(symptom)

            if previous_severity is not None and current_severity is not None:
                # Determine trend based on current severity compared to previous severity
                trend = 'improving' if current_severity < previous_severity else 'worsening'

                # Calculate variance and other metrics
                severities = [v[1] for v in values] + [current_severity]
                trend_analysis[symptom] = {
                    'trend': trend,
                    'variance': np.var(severities),
                    'max_severity': max(severities),
                    'current': current_severity,
                    'history': values  
                }
            elif current_severity is not None:
                # If there's no historical data, can only report the current state
                trend_analysis[symptom] = {
                    'trend': 'new',
                    'variance': None,
                    'max_severity': current_severity,
                    'current': current_severity,
                    'history': []  
                }

        return trend_analysis



    def process_screening_with_history(self, patient_id: str, symptoms: Dict) -> Dict:
        """
        Process screening with historical context, handling both new and existing patients.
        
        Args:
            patient_id: Patient's unique identifier
            symptoms: Dictionary of current symptoms and their severities
            
        Returns:
            Dict containing predictions, analysis, and recommendations
        """
        try:
            # First prepare the current symptoms for ML model
            current_features = self._prepare_current_features(symptoms)
            
            # Get historical data if available
            historical_data = self.patient_db.get_patient_historical_data(patient_id)
            
            # Prepare the final feature set for ML model
            if historical_data:
                # Patient exists - incorporate historical data
                trend_analysis = self.analyze_symptom_trends(historical_data, symptoms)
                # Prepare features for ML model with historical context
                features = self._prepare_features_with_history(symptoms, trend_analysis)
            else:
                # New patient - use only current symptoms
                trend_analysis = {}
                features = current_features
            
            # Make prediction using ML model
            prediction_result = self._make_prediction(features)
            
            # Store the screening record
            screening_data = {
                'symptoms': symptoms,
                'prediction': prediction_result,
                'created_at': datetime.now()
            }
            
            # Store in database
            self.patient_db.create_screening_record(patient_id, screening_data)
            
            # Prepare context for agent analysis
            context = {
                'current_symptoms': symptoms,
                'historical_data': historical_data,
                'trend_analysis': trend_analysis,
                'predictions': prediction_result
            }

            # Get comprehensive analysis from agent
            agent_analysis = self.healthcare_agent.analyze_patient_case(context)

            # Prepare response
            response = {
                'prediction_result': prediction_result,
                'has_history': bool(historical_data),
                'agent_analysis': agent_analysis
            }
            
            # Add historical analysis if available
            if historical_data:
                response['historical_analysis'] = {
                    'trend_analysis': trend_analysis,
                    'previous_screenings': len(historical_data)
                }
            
            return response
            
        except Exception as e:
            print(f"Error processing screening for patient {patient_id}: {str(e)}")
            raise

    def _prepare_current_features(self, symptoms: Dict) -> pd.DataFrame:
        """
        Prepare feature vector from current symptoms only
        """
        features = pd.DataFrame(0, index=[0], columns=self.standard_symptoms + ['duration_days'])
        
        for symptom, severity in symptoms.items():
            if symptom in features.columns:
                features.loc[0, symptom] = severity
        
        return features

    def _prepare_features_with_history(self, current_symptoms: Dict, trend_analysis: Dict) -> pd.DataFrame:
        """Prepare feature vector incorporating historical trends"""
        features = pd.DataFrame(0, index=[0], columns=self.standard_symptoms + ['duration_days'])
        
        for symptom, severity in current_symptoms.items():
            if symptom in features.columns:
                features.loc[0, symptom] = severity
                
                # Adjust feature based on historical trend if available
                if symptom in trend_analysis:
                    trend = trend_analysis[symptom]
                    if trend['trend'] == 'worsening':
                        features.loc[0, symptom] *= 1.2  # Increase weight for worsening symptoms
                    elif trend['trend'] == 'improving':
                        features.loc[0, symptom] *= 0.8  # Decrease weight for improving symptoms
        
        return features
    

    def _make_prediction(self, features: pd.DataFrame) -> Dict:
        """
        Make prediction using the ML model
        """
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            prediction_probs = self.model.predict_proba(features_scaled)
            
            # Get top predictions sorted by probability
            predictions = list(zip(self.model.classes_, prediction_probs[0]))
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Format prediction result
            return {
                'top_predictions': [
                    {
                        'condition': pred[0],
                        'probability': float(pred[1]),
                        'confidence_level': 'high' if pred[1] > 0.7 else 'medium' if pred[1] > 0.4 else 'low'
                    }
                    for pred in predictions[:3]  # Top 3 predictions
                ],
                'primary_prediction': {
                    'condition': predictions[0][0],
                    'probability': float(predictions[0][1])
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise

 
