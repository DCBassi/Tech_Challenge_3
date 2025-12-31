import pandas as pd
from assistant.llm_loader import chat_with_model
from pathlib import Path

def get_patient_record(patient_id):
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        df = pd.read_csv(BASE_DIR / "data" / "patients.csv")
        df['id'] = df['id'].astype(str)
        record = df[df['id'] == str(patient_id)]
        return record.to_dict(orient='records')[0] if not record.empty else None
    except: return None

def create_qa_chain(vectorstore, patient_id=None):
    patient_data = get_patient_record(patient_id)

    def qa(query):
        query_lower = query.lower()
        
        # 1. Trava de Entrada (Input Guardrail) - Impede perguntas de tempo e gerais
        off_topic_inputs = ["time", "hour", "date", "weather", "recipe", "code", "movie", "song", "game","book","shopping","travel","joke"]
        if any(term in query_lower for term in off_topic_inputs):
            return {"result": "I am sorry, but I can only assist with clinical and medical inquiries. This topic is out of my scope.", "source_documents": [], "patient_context": patient_data}

        # 2. BYPASS para dados fixos (Mantido original)
        if patient_data:
            if "age" in query_lower:
                return {"result": f"The patient is {patient_data['age']} years old.", "source_documents": [], "patient_context": patient_data}
            if ("name" in query_lower or "who is" in query_lower) and "risk" not in query_lower:
                return {"result": f"The patient's name is {patient_data['name']}.", "source_documents": [], "patient_context": patient_data}

        # 3. RAG para Conhecimento Medico
        docs = vectorstore.similarity_search(query, k=1)
        medical_context = docs[0].page_content if docs else "No medical context found."
        source_url = docs[0].metadata.get("url", "Internal Source") if docs else "N/A"
        
        # Prompt Ultra-Restritivo em Ingles
        template = f"""SYSTEM: You are a STRICT Medical Assistant. 
        You are ONLY allowed to talk about medicine and the provided context.
        If the question is about time, general knowledge, or anything else, you MUST refuse.

        RULES:
        1. No prescriptions.
        2. No definitive diagnoses.
        3. Only use medical context.

        MEDICAL CONTEXT: {medical_context}
        SOURCE: {source_url}
        USER QUESTION: {query}

        ANSWER:"""
        
        answer = chat_with_model(template, max_new_tokens=450)

        return {
            "result": answer, 
            "source_documents": docs,
            "patient_context": patient_data
        }

    return qa