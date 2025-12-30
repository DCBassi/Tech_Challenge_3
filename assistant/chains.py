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
        
        # 1. BYPASS para dados fixos (CSV) - Resposta imediata e precisa
        if patient_data:
            if "age" in query_lower:
                return {"result": f"The patient is {patient_data['age']} years old.", "source_documents": [], "patient_context": patient_data}
            if ("name" in query_lower or "who is" in query_lower) and "risk" not in query_lower:
                return {"result": f"The patient's name is {patient_data['name']}.", "source_documents": [], "patient_context": patient_data}
            if "diagnosis" in query_lower and "what is" not in query_lower:
                return {"result": f"The patient's current diagnosis is {patient_data['diagnosis']}.", "source_documents": [], "patient_context": patient_data}

        # 2. RAG para Conhecimento Médico (FAISS)
        # Reduzimos k=1 para o modelo focar em apenas uma fonte e não cortar o texto
        docs = vectorstore.similarity_search(query, k=1)
        medical_context = docs[0].page_content if docs else "No specific medical info found."
        
        # Prompt mais forte para evitar que o modelo pare no meio
        template = f"""SYSTEM: Use the context below to answer the user question completely.
CONTEXT: {medical_context}
USER QUESTION: {query}
COMPLETE ANSWER:"""
        
        answer = chat_with_model(template, max_new_tokens=450)

        return {
            "result": answer, 
            "source_documents": docs,
            "patient_context": patient_data
        }

    return qa