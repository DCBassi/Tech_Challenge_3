import pandas as pd
import re
from assistant.llm_loader import chat_with_model
from pathlib import Path

def get_patient_record(patient_id):
    """Recupera os dados do paciente no CSV."""
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        csv_path = BASE_DIR / "data" / "patients.csv"
        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str)
        record = df[df['id'] == str(patient_id)]
        return record.to_dict(orient='records')[0] if not record.empty else None
    except Exception as e:
        print(f"❌ Erro ao ler CSV de pacientes: {e}")
        return None

def create_qa_chain(vectorstore, patient_id=None):
    # Carrega os dados do paciente uma vez no início da cadeia
    patient_data = get_patient_record(patient_id)

    def qa(query):
        query_lower = query.lower().strip()

        # --- 1. FILTRO DE SEGURANÇA (OFF-TOPIC) ---
        # Prioridade alta: bloqueia perguntas irrelevantes antes de processar dados.
        off_topic_terms = ["weather", "recipe", "movie", "joke", "code", "time", "shopping"]
        if any(term in query_lower for term in off_topic_terms):
            return {
                "result": "I am a dedicated clinical assistant. I only provide information regarding medical inquiries and patient records.",
                "source_documents": [],
                "patient_context": patient_data,
                "strategy": "off_topic_filter"
            }
        
        # --- 2. BYPASS ESTRUTURADO (DADOS DO PACIENTE) ---
        # Resposta imediata baseada no CSV se a pergunta for sobre o paciente carregado.
        if patient_data:
            # Busca por NOME (Protegendo para não interceptar "Who is at risk")
            # Combinação das lógicas: Regex robusto + verificação simples
            name_patterns = [r"\bname\b", r"\bwho is the patient\b", r"\bwho is he\b", r"\bwho is she\b"]
            if (any(re.search(p, query_lower) for p in name_patterns) or "who is" in query_lower) and "risk" not in query_lower:
                return {
                    "result": f"The patient's name is {patient_data['name']}.",
                    "source_documents": [],
                    "patient_context": patient_data,
                    "strategy": "csv_direct"
                }
            
            # Busca por IDADE
            if any(term in query_lower for term in ["age", "how old", "years old"]):
                return {
                    "result": f"The patient, {patient_data['name']}, is {patient_data['age']} years old.",
                    "source_documents": [],
                    "patient_context": patient_data,
                    "strategy": "csv_direct"
                }

            # Busca por STATUS ou DIAGNÓSTICO
            # Combina termos do HEAD ("diagnosis") com os novos ("status", "condition")
            status_terms = ["status", "condition", "my diagnosis", "patient's diagnosis", "patient status", "what is the diagnosis"]
            if any(term in query_lower for term in status_terms):
                return {
                    "result": f"According to the records, the patient's current status is '{patient_data['status']}' for the diagnosis of {patient_data['diagnosis']}.",
                    "source_documents": [],
                    "patient_context": patient_data,
                    "strategy": "csv_direct"
                }

        # --- 3. RAG (CONHECIMENTO MÉDICO GERAL) ---
        # Se não caiu nos filtros acima, é uma pergunta clínica para o FAISS.
        docs = vectorstore.similarity_search(query, k=1)
        medical_context = docs[0].page_content if docs else "No specific medical reference found."
        
        # Prompt que integra os dados do paciente com a resposta médica
        context_for_llm = f"The patient is {patient_data['name']}, {patient_data['age']} years old, with {patient_data['diagnosis']}." if patient_data else ""
        
        template = f"""SYSTEM: You are a STRICT Medical Assistant. 
Answer the user's question based ONLY on the Medical Reference provided.
If the patient's context is relevant (like age or current diagnosis), mention it.

PATIENT CONTEXT: {context_for_llm}
MEDICAL REFERENCE: {medical_context}
USER QUESTION: {query}

FINAL ANSWER (Be objective and clinical):"""
        
        answer = chat_with_model(template, max_new_tokens=450)

        return {
            "result": answer, 
            "source_documents": docs,
            "patient_context": patient_data,
            "strategy": "rag_llm"
        }

    return qa