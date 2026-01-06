import pandas as pd
import re
from assistant.llm_loader import chat_with_model
from pathlib import Path

def get_patient_record(patient_id):
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        csv_path = BASE_DIR / "data" / "patients.csv"
        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str)
        record = df[df['id'] == str(patient_id)]
        return record.to_dict(orient='records')[0] if not record.empty else None
    except: return None

def create_qa_chain(vectorstore, patient_id=None):
    patient_data = get_patient_record(patient_id)

    def qa(query):
        query_lower = query.lower().strip()
        
        # 1. FILTRO GERAL
        off_topic = ["movie", "recipe", "weather", "song", "game", "joke"]
        if any(term in query_lower for term in off_topic):
            return {"result": "I assist only with medical inquiries related to the patient's condition.", "source_documents": [], "patient_context": patient_data}

        # 2. BYPASS CSV (Dados Estáticos)
        if patient_data:
            if any(x in query_lower for x in ["name", "who is"]):
                return {"result": f"The patient's name is {patient_data['name']}.", "source_documents": [], "patient_context": patient_data}
            if "age" in query_lower and len(query_lower.split()) < 6:
                return {"result": f"The patient is {patient_data['age']} years old.", "source_documents": [], "patient_context": patient_data}
            if "status" in query_lower and len(query_lower.split()) < 6:
                return {"result": f"The status of {patient_data['name']} is {patient_data['status']}.", "source_documents": [], "patient_context": patient_data}

        # 3. FILTRO DE AFINIDADE (Segurança Universal)
        diag_full = patient_data['diagnosis'].lower() if patient_data else ""
        ignore_terms = {"cancer", "stage", "of", "the", "and", "type", "ii", "iii", "iv", "i"}
        diag_keywords = [w for w in re.findall(r'\w+', diag_full) if w not in ignore_terms and len(w) > 2]

        medical_triggers = ["cancer", "tumor", "leukemia", "glioma", "astrocytoma", "vulvar", "urethral", "prostate"]
        if any(mt in query_lower for mt in medical_triggers):
            if not any(dk in query_lower for dk in diag_keywords):
                return {"result": f"Access Denied. I can only discuss information related to {patient_data['diagnosis']}.", "source_documents": [], "patient_context": patient_data}

        # 4. RAG DETERMINÍSTICO
        docs = vectorstore.similarity_search(query, k=1)
        medical_context = docs[0].page_content if docs else ""
        
        # Prompt enxuto para evitar confusão no modelo
        template = f"""SYSTEM: You are a concise medical assistant. 
        Answer ONLY based on the Reference below. Max 2 sentences.
        If not found, say you only assist with {patient_data['diagnosis']}.

        REFERENCE: {medical_context[:400]}

        QUESTION: {query}
        ANSWER:"""
        
        answer = chat_with_model(template, max_new_tokens=120)
        
        # Fallback de segurança se o modelo negar a resposta correta
        if ("sorry" in answer.lower() or "assist" in answer.lower()) and any(dk in query_lower for dk in diag_keywords):
             answer = f"According to clinical records for {patient_data['diagnosis']}: {medical_context[:250]}..."

        return {"result": answer, "source_documents": docs, "patient_context": patient_data}

    return qa