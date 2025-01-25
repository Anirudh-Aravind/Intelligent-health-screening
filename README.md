# 🏥 AI-Powered Healthcare Diagnosis System

## Project Overview

This project represents an innovative approach to healthcare screening, leveraging synthetic data generation, machine learning, and agentic AI to provide intelligent medical diagnostics. The core mission is to demonstrate how advanced AI technologies can enhance medical screening and decision support processes.


## Project Demonstration

You can try the app by following this [link](https://github.com/Anirudh-Aravind/Intelligent-health-screening/raw/main/medical_screening.mp4).

## Project Demo

![Video Demo](https://github.com/Anirudh-Aravind/Intelligent-health-screening/raw/main/medical_screening.mp4)

 You can try the app by following this link: https://heatlhcareassist.streamlit.app/


## 🌟 Key Highlights

- **Synthetic Data Generation**: Unique approach to creating realistic medical datasets
- **Machine Learning Diagnosis Prediction**: Advanced RandomForest classifier
- **Agentic AI Analysis**: Multi-agent system for comprehensive medical insights
- **Built Entirely with Free Resources**: Proving high-quality solutions can be developed cost-effectively

## Installation

### Prerequisites

- Python 3.8+
- MySQL (local or cloud-based)
- API Key for Groq (for multi-agent AI processing)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/healthcare-project.git
cd healthcare-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Environment Variables

Create a `.env` file in the project root with the following configurations:
```
MYSQL_HOST=localhost
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_PORT=3306
MYSQL_DATABASE=healthcare_db
GROQ_API_KEY=your_groq_api_key
ENCRYPTION_KEY=your_generated_fernet_key
```

## Project Structure

### 1. `disease_prediction_model_training.py`
- Generates synthetic medical data
- Trains RandomForest classifier
- Creates probabilistic disease prediction model

### 2. `healthcare_system.py`
- Central orchestration of healthcare screening
- Coordinates model predictions and patient data processing

### 3. `multi_agent.py`
- **Agentic AI Core**: Most innovative component
- Uses Groq's Mixtral 8x7B model
- Implements multi-agent analysis strategy
- Breaks down complex medical data into manageable chunks
- Provides comprehensive medical insights

### 4. `patient_db_operations.py`
- Secure patient data management
- Encrypted database interactions
- Supports patient registration and historical data tracking

### 5. `streamlit_ui.py`
- Interactive web interface
- Enables patient screening and result visualization

## 🚀 Agentic AI Implementation

The multi-agent system is the project's breakthrough innovation:

- **Chunk-based Processing**: Handles large, complex medical datasets
- **Multiple Analysis Chains**:
  - Symptom Analysis
  - Treatment Recommendations
  - Risk Assessment
- **Adaptive Intelligence**: Dynamically adjusts analysis based on input context

### Sample Agentic Analysis Workflow
1. Receive patient symptoms
2. Break down data into processable chunks
3. Apply specialized analysis chains
4. Synthesize comprehensive medical insights

## 🔧 Technological Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **AI Processing**: Groq Mixtral 8x7B
- **Database**: MySQL
- **UI**: Streamlit
- **Encryption**: Cryptography (Fernet)

## 💡 Performance Considerations

- Synthetic data generation ensures robust model training
- Model complexity balanced with computational efficiency
- Chunk-based AI processing prevents overwhelming large language models

## Deployment Options

### Local Deployment
- Use localhost MySQL
- Run Streamlit: `streamlit run streamlit_ui.py`

### Cloud Deployment
Recommended Cloud Services (Free/Low-Cost):
- **Database**: 
  - Clever Cloud MySQL
  - AWS RDS (Free tier)
  - Google Cloud SQL
- **Hosting**: 
  - Heroku
  - PythonAnywhere
  - Streamlit Cloud

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request


## 🌈 Final Note

This project demonstrates that cutting-edge AI solutions in healthcare can be developed using free resources, focusing on functionality and intelligent design rather than expensive infrastructure.

**Disclaimer**: Not a substitute for professional medical advice. For medical concerns, always consult healthcare professionals.
