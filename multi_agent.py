import os
from datetime import datetime
from typing import Dict, Optional, Union
import ast
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class HealthcareMultiAgentSystem:
    def __init__(self):
        """Initialize the agent system with Groq"""
        self.api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name="mixtral-8x7b-32768"
        )
        self.max_chunk_size = 1024  # Adjust based on model's token limit
        self.setup_chains()

    def setup_chains(self):
        """Setup analysis chains for each type of analysis"""
        self.symptom_chain = self._create_analysis_chain("""
            Analyze these symptoms and provide insights:
            {data}
            
            Include:
            1. Primary symptoms and severity
            2. Symptom interactions
            3. Health impact assessment
        """)
        
        self.trend_chain = self._create_analysis_chain("""
            Analyze these health trends:
            {data}
            
            Include:
            1. Key trends
            2. Pattern identification
            3. Notable changes
        """)
        
        self.treatment_chain = self._create_analysis_chain("""
            Recommend treatments based on:
            {data}
            
            Include:
            1. Treatment recommendations
            2. Lifestyle changes
            3. Follow-up needs
        """)
        
        self.risk_chain = self._create_analysis_chain("""
            Assess health risks for:
            {data}
            
            Include:
            1. Risk factors
            2. Severity levels
            3. Mitigation steps
        """)

    def _create_analysis_chain(self, template: str):
        """Create a chain with the new LangChain syntax"""
        prompt = PromptTemplate(
            input_variables=["data"],
            template=template
        )
        return (
            {"data": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _serialize_data(self, data: Union[Dict, str]) -> str:
        """Safely serialize data to JSON string"""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, cls=DateTimeEncoder)
        except TypeError as e:
            return str(data)

    def _deserialize_data(self, data_str: str) -> Union[Dict, str]:
        """Safely deserialize JSON string back to dict"""
        if not isinstance(data_str, str):
            return data_str
        try:
            return json.loads(data_str)
        except:
            try:
                return ast.literal_eval(data_str)
            except:
                return data_str

    def _chunk_data(self, data: Dict) -> list:
        """Break down data into manageable chunks"""
        chunks = []
        
        # First, serialize the entire data
        data_str = self._serialize_data(data)
        
        # Convert serialized string back to dictionary for proper chunking
        data_dict = self._deserialize_data(data_str)
        
        if len(data_str) <= self.max_chunk_size:
            return [data_dict]
            
        # Handle dictionary data
        if isinstance(data_dict, dict):
            current_chunk = {}
            current_size = 0
            
            for key, value in data_dict.items():
                item = {key: value}
                item_str = self._serialize_data(item)
                item_size = len(item_str)
                
                if item_size > self.max_chunk_size:
                    # If single item is too large, needs further chunking
                    if isinstance(value, list):
                        # Chunk lists into smaller pieces
                        for i in range(0, len(value), 5):  # Process 5 items at a time
                            sub_chunk = {key: value[i:i+5]}
                            sub_chunk_str = self._serialize_data(sub_chunk)
                            if len(sub_chunk_str) <= self.max_chunk_size:
                                chunks.append(sub_chunk)
                    elif isinstance(value, dict):
                        # Recursively chunk nested dictionaries
                        sub_chunks = self._chunk_data(value)
                        chunks.extend([{key: sc} for sc in sub_chunks])
                else:
                    if current_size + item_size <= self.max_chunk_size:
                        current_chunk.update(item)
                        current_size += item_size
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = item
                        current_size = item_size
            
            if current_chunk:
                chunks.append(current_chunk)
                
        return chunks if chunks else [data_dict]

    def _process_chunks(self, chunks: list, analysis_func) -> str:
        """Process data chunks and combine results"""
        results = []
        for chunk in chunks:
            try:
                chunk_str = self._serialize_data(chunk)
                if len(chunk_str) <= self.max_chunk_size:
                    result = analysis_func(chunk_str)
                    results.append(result)
            except Exception as e:
                results.append(f"Error processing chunk: {str(e)}")
        
        return "\n".join(results)

    def _analyze_symptoms(self, context: Union[Dict, str]) -> str:
        """Analyze symptoms with chunk handling"""
        try:
            if isinstance(context, str):
                context = self._deserialize_data(context)
            symptoms_data = context.get('current_symptoms', {})
            chunks = self._chunk_data(symptoms_data)
            return self._process_chunks(chunks, 
                lambda x: self.symptom_chain.invoke(x))
        except Exception as e:
            return f"Error analyzing symptoms: {str(e)}"

    def _analyze_trends(self, context: Union[Dict, str]) -> str:
        """Analyze trends with chunk handling"""
        try:
            if isinstance(context, str):
                context = self._deserialize_data(context)
            historical_data = context.get('historical_data', {})
            chunks = self._chunk_data(historical_data)
            return self._process_chunks(chunks, 
                lambda x: self.trend_chain.invoke(x))
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"

    def _provide_treatment_recommendations(self, context: Union[Dict, str]) -> str:
        """Provide treatments with chunk handling"""
        try:
            if isinstance(context, str):
                context = self._deserialize_data(context)
            chunks = self._chunk_data(context)
            return self._process_chunks(chunks, 
                lambda x: self.treatment_chain.invoke(x))
        except Exception as e:
            return f"Error providing treatment recommendations: {str(e)}"

    def _assess_risks(self, context: Union[Dict, str]) -> str:
        """Assess risks with chunk handling"""
        try:
            if isinstance(context, str):
                context = self._deserialize_data(context)
            chunks = self._chunk_data(context)
            return self._process_chunks(chunks, 
                lambda x: self.risk_chain.invoke(x))
        except Exception as e:
            return f"Error assessing risks: {str(e)}"

    def analyze_patient_case(self, context: Dict) -> Dict:
        """Main analysis function with chunk handling"""
        try:
            # Convert context to proper format for analysis
            context_str = self._serialize_data(context)
            context_dict = self._deserialize_data(context_str)
            
            # Chunk the initial context for comprehensive analysis
            chunks = self._chunk_data(context_dict)
            initial_analyses = []
            
            for chunk in chunks:
                try:
                    chunk_str = self._serialize_data(chunk)
                    if len(chunk_str) <= self.max_chunk_size:
                        analysis = self.llm.invoke(
                            f"""Analyze this patient data:
                            Context: {chunk_str}
                            
                            Provide analysis of:
                            1. Symptoms
                            2. Trends (if historical data present)
                            3. Treatment needs
                            4. Risks
                            """
                        ).content
                        initial_analyses.append(analysis)
                except Exception as e:
                    initial_analyses.append(f"Error processing chunk: {str(e)}")

            # Compile detailed analysis
            detailed_analysis = {
                'symptom_analysis': self._analyze_symptoms(context_dict),
                'treatment_recommendations': self._provide_treatment_recommendations(context_dict),
                'risk_assessment': self._assess_risks(context_dict),
                'agent_summary': "\n".join(initial_analyses)
            }
            
            if context_dict.get('historical_data'):
                detailed_analysis['trend_analysis'] = self._analyze_trends(context_dict)

            return detailed_analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }