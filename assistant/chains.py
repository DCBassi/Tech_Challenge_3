import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from assistant.llm_loader import model, tokenizer, chat_with_model
from pathlib import Path

# =========================
# 1. DADOS ESTRUTURADOS (CONSULTA CSV)
# =========================
def get_patient_record(patient_id):
    try:
        # Define o caminho para o CSV
        BASE_DIR = Path(__file__).resolve().parent.parent
        csv_path = BASE_DIR / "data" / "patients.csv"
        
        # Carrega o "banco de dados"
        df = pd.read_csv(csv_path)
        
        # Converte o ID para string para garantir a comparação
        df['id'] = df['id'].astype(str)
        
        # Busca o paciente pelo ID
        record = df[df['id'] == str(patient_id)]
        
        if not record.empty:
            # Retorna o registro formatado como dicionário
            return record.to_dict(orient='records')[0]
        else:
            return f"Patient ID {patient_id} not found in database."
            
    except Exception as e:
        return f"Error accessing structured database: {e}"

# =========================
# 2. CONFIGURAÇÃO DA PIPELINE LANGCHAIN
# =========================
def get_langchain_llm():
    # Integra seu modelo local ao ecossistema LangChain
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        temperature=0.01,
        repetition_penalty=1.15,
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=hf_pipe)

# =========================
# 3. CONTEXTUALIZAÇÃO E QA CHAIN
# =========================
def create_qa_chain(vectorstore, patient_id=None):
    """
    Retorna uma função que integra RAG (FAISS) + Dados Estruturados (Patient Record)
    """
    llm = get_langchain_llm()
    
    # Busca o prontuário se um ID for fornecido
    patient_data = get_patient_record(patient_id) if patient_id else "No patient context provided."

    def qa(query):
        # A. Busca literatura médica (RAG)
        docs = vectorstore.similarity_search(query, k=3)
        medical_context = "\n".join([doc.page_content for doc in docs])

        # B. Template com Contextualização do Paciente
        # Aqui unimos o requisito de dados estruturados + LLM customizada
        # No chains.py
        template = f"""### Instruction:
        You are a medical specialist assistant. Use the PATIENT RECORD and MEDICAL CONTEXT below to answer the user.
        Answer specifically for the patient {patient_data.get('name', 'the patient')}, who is {patient_data.get('age')} years old and diagnosed with {patient_data.get('diagnosis')}.

        ### PATIENT RECORD:
        {patient_data}

        ### MEDICAL CONTEXT:
        {medical_context}

        ### QUESTION:
        {query}

### ANSWER:
"""
        
        # C. Execução via Pipeline Customizada
        # Usamos o chat_with_model ou chamamos a pipeline do langchain
        # Para garantir consistência com seu treino, vamos usar o chat_with_model diretamente:
        answer = chat_with_model([{"role": "user", "content": template}])

        return {
            "result": answer, 
            "source_documents": docs,
            "patient_context": patient_data
        }

    return qa