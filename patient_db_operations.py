from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List, Any
import json
import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

class PatientDatabase:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database connection
        self.engine = create_engine(
            f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
            f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DATABASE')}"
        )

        # Initialize encryption
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        self.cipher_suite = Fernet(self.encryption_key)

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve patient information from database
        
        Args:
            patient_id: Unique patient identifier
            
        Returns:
            Dict containing patient information if found, None otherwise
        """
        try:
            with Session(self.engine) as session:
                query = text("""
                    SELECT p.*, 
                           MAX(sr.created_at) as last_screening_date,
                           COUNT(sr.id) as screening_count
                    FROM patients p
                    LEFT JOIN screening_records sr ON p.patient_id = sr.patient_id
                    WHERE p.patient_id = :patient_id
                    GROUP BY p.patient_id
                """)
                
                result = session.execute(query, {"patient_id": patient_id}).mappings().first()
                
                if result is None:
                    return None
                
                # Decrypt sensitive information
                demographics = json.loads(
                    self.cipher_suite.decrypt(result['demographics'].encode()).decode()
                )
                medical_history = json.loads(
                    self.cipher_suite.decrypt(result['medical_history'].encode()).decode()
                )
                
                return {
                    "patient_id": result['patient_id'],
                    "demographics": demographics,
                    "medical_history": medical_history,
                    "created_at": result['created_at'],
                    "last_screening_date": result['last_screening_date'],
                    "screening_count": result['screening_count']
                }
        except Exception as e:
            self.logger.error(f"Database error in get_patient: {str(e)}")
            raise Exception("Error accessing patient database")

    def register_patient(self, patient_data: Dict[str, Any]) -> str:
        """
        Register a new patient in the database
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Newly created patient_id
        """
        try:
            with Session(self.engine) as session:
                # Check if patient with similar details exists
                existing_patient = self._check_existing_patient(session, patient_data)
                if existing_patient:
                    raise Exception("Patient with similar details already exists")
                
                # Generate unique patient ID
                patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Encrypt sensitive information
                encrypted_demographics = self.cipher_suite.encrypt(
                    json.dumps(patient_data).encode()
                ).decode()
                
                encrypted_medical_history = self.cipher_suite.encrypt(
                    json.dumps({
                        "conditions": patient_data.get('medical_history', {}).get('conditions', []),
                        "allergies": patient_data.get('medical_history', {}).get('allergies', ''),
                        "medications": patient_data.get('medical_history', {}).get('medications', '')
                    }).encode()
                ).decode()
                
                # Insert new patient record
                query = text("""
                    INSERT INTO patients 
                    (patient_id, demographics, medical_history, created_at, updated_at)
                    VALUES 
                    (:patient_id, :demographics, :medical_history, :created_at, :updated_at)
                """)
                
                session.execute(query, {
                    "patient_id": patient_id,
                    "demographics": encrypted_demographics,
                    "medical_history": encrypted_medical_history,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                })
                
                # Create initial screening record if symptoms provided
                if 'symptoms' in patient_data:
                    self._create_initial_screening(session, patient_id, patient_data['symptoms'])
                
                session.commit()
                return patient_id
        except Exception as e:
            self.logger.error(f"Database error in register_patient: {str(e)}")
            raise Exception(f"Error registering patient: {str(e)}")

    def _check_existing_patient(self, session: Session, patient_data: Dict) -> bool:
        """
        Check if patient with similar details exists
        Returns True if similar patient found, False otherwise
        """
        try:
            query = text("SELECT demographics FROM patients")
            results = session.execute(query).mappings()
            
            for row in results:
                existing_demographics = json.loads(
                    self.cipher_suite.decrypt(row['demographics'].encode()).decode()
                )
                
                if (
                    existing_demographics.get('name') == patient_data.get('name') and
                    existing_demographics.get('age') == patient_data.get('age') and
                    existing_demographics.get('contact', {}).get('phone') == 
                    patient_data.get('contact', {}).get('phone')
                ):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error in _check_existing_patient: {str(e)}")
            return False

    def _create_initial_screening(self, session: Session, patient_id: str, symptoms: Dict):
        """Create initial screening record for new patient"""
        try:
            encrypted_symptoms = self.cipher_suite.encrypt(
                json.dumps(symptoms).encode()
            ).decode()
            
            query = text("""
                INSERT INTO screening_records 
                (patient_id, symptoms, created_at)
                VALUES 
                (:patient_id, :symptoms, :created_at)
            """)
            
            session.execute(query, {
                "patient_id": patient_id,
                "symptoms": encrypted_symptoms,
                "created_at": datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error creating initial screening: {str(e)}")

    def get_patient_historical_data(self, patient_id: str, timeframe_days: int = 90) -> List[Dict]:
        """
        Retrieve patient's historical screening data within specified timeframe
        
        Args:
            patient_id: Patient's unique identifier
            timeframe_days: Number of days to look back
            
        Returns:
            List of dictionaries containing historical screening data
        """
        try:
            with Session(self.engine) as session:
                cutoff_date = (datetime.now() - timedelta(days=timeframe_days))
                
                query = text("""
                    SELECT symptoms, prediction, created_at, actual_diagnosis 
                    FROM screening_records 
                    WHERE patient_id = :patient_id AND created_at >= :cutoff_date 
                    ORDER BY created_at DESC
                """)
                
                results = session.execute(query, {
                    "patient_id": patient_id,
                    "cutoff_date": cutoff_date
                }).mappings()
                
                historical_data = []
                for record in results:
                    historical_data.append({
                        'symptoms': json.loads(self.cipher_suite.decrypt(record['symptoms'].encode())),
                        'prediction': json.loads(self.cipher_suite.decrypt(record['prediction'].encode())),
                        'created_at': record['created_at'],
                        'actual_diagnosis': record['actual_diagnosis']
                    })
                
                return historical_data
        except Exception as e:
            self.logger.error(f"Error retrieving historical data for patient {patient_id}: {str(e)}")
            raise

    def create_screening_record(self, patient_id: str, screening_data: Dict) -> int:
        """
        Create a new screening record in the database.
        
        Args:
            patient_id (str): The unique identifier of the patient
            screening_data (Dict): Dictionary containing:
                - symptoms (Dict): Dictionary of symptoms and their severities
                - prediction (Dict): Model predictions and confidence scores
                - actual_diagnosis (str, optional): Actual diagnosis if available
                - created_at (datetime, optional): Timestamp of screening
        
        Returns:
            int: ID of the newly created screening record
        """
        try:
            with Session(self.engine) as session:
                if 'symptoms' not in screening_data or 'prediction' not in screening_data:
                    raise ValueError("Screening data must contain 'symptoms' and 'prediction'")
                
                encrypted_symptoms = self.cipher_suite.encrypt(
                    json.dumps(screening_data['symptoms']).encode('utf-8')
                ).decode()

                encrypted_prediction = self.cipher_suite.encrypt(
                    json.dumps(screening_data['prediction']).encode('utf-8')
                ).decode()
                
                query = text("""
                    INSERT INTO screening_records 
                        (patient_id, symptoms, prediction, created_at)
                    VALUES 
                        (:patient_id, :symptoms, :prediction, :created_at)
                """)
                
                session.execute(query, {
                    "patient_id": patient_id,
                    "symptoms": encrypted_symptoms,
                    "prediction": encrypted_prediction,
                    "created_at": screening_data.get('created_at', datetime.now())
                })
                
                session.commit()
                self.logger.info(f"Created screening record for patient {patient_id}")
        except Exception as e:
            self.logger.error(f"Error creating screening record: {str(e)}")
            raise
