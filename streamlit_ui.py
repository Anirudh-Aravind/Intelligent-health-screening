import streamlit as st
import pandas as pd
from datetime import datetime
import time
import plotly.graph_objects as go
from healthcare_system import HealthcareSystem  
from patient_db_operations import PatientDatabase
import re

class HealthcareUI:
    def __init__(self):
        self.health_system = HealthcareSystem()
        self.patient_db = PatientDatabase()
        st.set_page_config(
            page_title="Healthcare Screening System",
            page_icon="🏥",
            layout="wide"
        )
        

    def validate_patient_id(self, patient_id: str) -> bool:
        """Validate patient ID format"""
        return bool(re.match(r'^P\d{14}$', patient_id))

    def run(self):
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["Home", "Patient Registration", "Patient Screening"]
        )

        if page == "Home":
            self.show_home_page()
        elif page == "Patient Registration":
            self.show_registration_page()
        elif page == "Patient Screening":
            self.show_screening_page()

    def show_home_page(self):
        st.title("🏥 Healthcare Screening System")
        st.markdown("""
        ### Welcome to the Healthcare Screening System
        
        This system helps healthcare providers:
        * Check patient records
        * Register new patients
        * Conduct health screenings
        * Get AI-powered diagnosis predictions
        
        Please use the sidebar to navigate to different sections.
        """)

        # Patient ID checker
        st.header("Quick Patient Lookup")
        with st.form("patient_lookup"):
            lookup_id = st.text_input("Enter Patient ID")
            submit_lookup = st.form_submit_button("Check Patient")

            if submit_lookup and lookup_id:
                if self.validate_patient_id(lookup_id):
                    patient_data = self.patient_db.get_patient(lookup_id)
                    if patient_data:
                        st.success("Patient found in the system!")
                        # Display basic non-sensitive info
                        demographics = patient_data['demographics']
                        medical_history = patient_data['medical_history']
                        st.write(f"**Patient Name:** {demographics['name']}")
                        st.write(f"**Patient Age:** {demographics['age']}")
                        # Display Patient Medical History with structured formatting
                        st.write("##### Patient Medical History:")

                        # Display Conditions
                        st.write("**Conditions**:")
                        if medical_history['conditions']:
                            for condition in medical_history['conditions']:
                                st.write(f"- {condition}")
                        else:
                            st.write("- None")

                        # Display Allergies
                        st.write("**Allergies**:")
                        if medical_history['allergies']:
                            st.write(f"- {medical_history['allergies']}")
                        else:
                            st.write("- None")

                        # Display Medications
                        st.write("**Medications**:")
                        if medical_history['medications']:
                            st.write(f"- {medical_history['medications']}")
                        else:
                            st.write("- None")
                        st.write(f"**Last Screening:** {patient_data['last_screening_date']}")
                    else:
                        st.warning("Patient not found. Please register them in the Patient Registration page.")
                else:
                    st.error("Invalid Patient ID format. ID should start with 'P' followed by 14 digits.")

    def show_registration_page(self):
        st.title("Patient Registration")
        
        with st.form("registration_form"):
            st.header("Patient Information")
            
            # Basic information
            name = st.text_input("Full Name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120)
                gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
            with col2:
                phone = st.text_input("Phone Number")
                email = st.text_input("Email (optional)")

            # Medical history
            st.header("Basic Medical History")
            existing_conditions = st.multiselect(
                "Existing Medical Conditions",
                ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "None"]
            )
            allergies = st.text_area("Allergies (if any)")
            medications = st.text_area("Current Medications (if any)")

            # Submit button
            submitted = st.form_submit_button("Register Patient")

            if submitted:
                if not name or not age or gender == "Select" or not phone:
                    st.error("Please fill all required fields.")
                else:
                    # Prepare patient data
                    patient_data = {
                        'name': name,
                        'age': age,
                        'gender': gender,
                        'contact': {
                            'phone': phone,
                            'email': email
                        },
                        'medical_history': {
                            'conditions': existing_conditions,
                            'allergies': allergies,
                            'medications': medications
                        }
                    }

                    # Register patient
                    try:
                        patient_id = self.patient_db.register_patient(patient_data)
                        st.success(f"""
                        Patient registered successfully!
                        Patient ID: {patient_id}
                        Please save this ID for future reference.
                        """)
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")

    def show_screening_page(self):
        st.title("Patient Screening")
        
        # Patient ID input
        patient_id = st.text_input("Enter Patient ID")
        
        if patient_id:
            patient_data = self.patient_db.get_patient(patient_id)
            if patient_data:
                st.success("Patient found!")
                demographics = patient_data['demographics']
                st.write(f"Patient Name: {demographics['name']}")
                st.write(f"Age: {demographics['age']}")

                # Symptom input form
                with st.form("screening_form"):
                    st.header("Symptom Assessment")
                    
                    # Create symptom severity inputs
                    symptoms = {}
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for symptom in self.health_system.standard_symptoms[:8]:
                            symptoms[symptom] = st.slider(
                                f"{symptom.replace('_', ' ').title()}",
                                0, 3, 0,
                                help="0: None, 1: Mild, 2: Moderate, 3: Severe"
                            )
                    
                    with col2:
                        for symptom in self.health_system.standard_symptoms[8:]:
                            symptoms[symptom] = st.slider(
                                f"{symptom.replace('_', ' ').title()}",
                                0, 3, 0,
                                help="0: None, 1: Mild, 2: Moderate, 3: Severe"
                            )
                    
                    duration = st.number_input("Duration (days)", min_value=1, max_value=30, value=1)
                    symptoms['duration_days'] = duration

                    submitted = st.form_submit_button("Process Screening")

                    if submitted:
                        # Show processing message
                        with st.spinner('Processing screening...'):
                            # Filter out symptoms with severity 0
                            active_symptoms = {k: v for k, v in symptoms.items() if v > 0 or k == 'duration_days'}
                            
                            # Process screening with history
                            result = self.health_system.process_screening_with_history(patient_id, active_symptoms)
                            
                            # Display results
                            if result:
                                self.display_current_symptoms(active_symptoms)
                                self.display_screening_results(result)

            else:
                st.error("Patient not found. Please register the patient first.")


    def display_screening_results(self, response: dict):
        """Display medical screening results in Streamlit UI"""
        
        st.header("Screening Results")
        
        # Create tabs for different sections
        tabs = st.tabs(["Predictions", "Analysis", "Historical Data"])
        
        # Predictions Tab
        with tabs[0]:
            st.subheader("Prediction Results")
            
            # Display top predictions in a nice table
            predictions_df = pd.DataFrame(response['prediction_result']['top_predictions'])
            
            # Style the probability as percentage
            predictions_df['probability'] = predictions_df['probability'].apply(lambda x: f"{x*100:.1f}%")
            
            # Create colored boxes for confidence levels using a proper styling function
            def style_confidence_levels(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                colors = {
                    'high': 'background-color: #90EE90',
                    'medium': 'background-color: #FFE4B5',
                    'low': 'background-color: #FFB6C1'
                }
                
                for idx in df.index:
                    styles.loc[idx, 'confidence_level'] = colors.get(df.loc[idx, 'confidence_level'], '')
                
                return styles
            
            # Apply the styling
            styled_df = predictions_df.style.apply(style_confidence_levels, axis=None)
            
            # Display the styled dataframe
            st.dataframe(styled_df, hide_index=True)
            
            # Primary prediction highlight
            st.info(f"Primary Prediction: {response['prediction_result']['primary_prediction']['condition']} "
                    f"(Probability: {response['prediction_result']['primary_prediction']['probability']*100:.1f}%)")
        
        # Analysis Tab
        with tabs[1]:
            st.subheader("Agent Analysis")
            
            if 'agent_analysis' in response:
                # Create expandable sections for each analysis component
                if 'symptom_analysis' in response['agent_analysis']:
                    with st.expander("Symptom Analysis", expanded=True):
                        st.write(response['agent_analysis']['symptom_analysis'])
                
                if 'treatment_recommendations' in response['agent_analysis']:
                    with st.expander("Treatment Recommendations"):
                        st.write(response['agent_analysis']['treatment_recommendations'])
                
                if 'risk_assessment' in response['agent_analysis']:
                    with st.expander("Risk Assessment"):
                        st.write(response['agent_analysis']['risk_assessment'])
                
                if 'agent_summary' in response['agent_analysis']:
                    st.success("Agent Summary")
                    st.write(response['agent_analysis']['agent_summary'])
            else:
                st.warning("Agent analysis not available. This might be due to processing errors.")
                
                # Display any error information if available
                if 'error' in response:
                    st.error(f"Error during analysis: {response['error']}")
        
        # Historical Data Tab
        with tabs[2]:
            if response.get('has_history'):
                st.subheader("Historical Analysis")
                
                # Display trend analysis if available
                if 'historical_analysis' in response and 'trend_analysis' in response['historical_analysis']:
                    trend_data = response['historical_analysis']['trend_analysis']
                    
                    # Create columns for metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Previous Screenings", 
                            response['historical_analysis'].get('previous_screenings', 0)
                        )
                    with col2:
                        st.metric(
                            "Active Symptoms", 
                            len(trend_data) if trend_data else 0
                        )
                    
                    # Display trend details
                    if trend_data:
                        st.subheader("Symptom Trends")
                        for symptom, data in trend_data.items():
                            if isinstance(data, dict):  # Ensure data is a dictionary
                                trend_direction = "📈" if data.get('trend') == 'worsening' else "📉"
                                trend_color = "red" if data.get('trend') == 'worsening' else "green"
                                
                                st.markdown(
                                    f"**{symptom}** {trend_direction} "
                                    f"<span style='color:{trend_color}'>{data.get('trend', 'unknown')}</span>",
                                    unsafe_allow_html=True
                                )
                                
                                # Create metrics for the symptom
                                mcol1, mcol2 = st.columns(2)
                                with mcol1:
                                    st.metric("Current Severity", data.get('current', 'N/A'))
                                with mcol2:
                                    st.metric("Max Severity", data.get('max_severity', 'N/A'))
            else:
                st.info("No historical data available for this patient.")
                
        # Timestamp information
        st.sidebar.info(f"Last Updated: {datetime.fromisoformat(response['prediction_result']['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")

    def display_current_symptoms(self, symptoms: dict):
        """Display current symptoms in a separate section"""
        st.header("Current Symptoms")
        
        # Create a DataFrame for better visualization
        symptoms_df = pd.DataFrame([
            {"Symptom": symptom, "Severity": severity}
            for symptom, severity in symptoms.items()
        ])
        
        # Create a bar chart for symptom severities
        fig = go.Figure(data=[
            go.Bar(
                x=symptoms_df['Symptom'],
                y=symptoms_df['Severity'],
                marker_color=['#FF9999' if severity > 7 else '#99FF99' if severity < 4 else '#FFCC99' 
                            for severity in symptoms_df['Severity']]
            )
        ])
        
        fig.update_layout(
            title="Symptom Severity Chart",
            xaxis_title="Symptoms",
            yaxis_title="Severity (0-10)",
            yaxis_range=[0, 10]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed symptoms table
        st.dataframe(symptoms_df, hide_index=True)


if __name__ == "__main__":
    ui = HealthcareUI()
    ui.run()