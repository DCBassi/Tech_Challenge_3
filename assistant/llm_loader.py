# llm_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import json
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# =========================
# CONFIGURAÇÃO DO MODELO
# =========================
BASE_PATH = Path(__file__).parent.parent
MODEL_DIR = (BASE_PATH / "models" / "llama_finetuned").resolve()

if not MODEL_DIR.exists():
    MODEL_DIR = (BASE_PATH / "models").resolve()

MODEL_PATH = str(MODEL_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
    print(f"❌ ERRO: Pasta do modelo vazia ou inexistente.")
else:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            device_map="auto", 
            local_files_only=True,
            torch_dtype=torch.float32 
        )
        model.eval()
        print(f"✅ Modelo carregado com sucesso no {DEVICE}")
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")

# =========================
# FUNÇÃO DE CHAT COM O MODELO (VERSÃO FINAL)
# =========================
def chat_with_model(messages, max_new_tokens=400, temperature=0.01):
    if model is None or tokenizer is None:
        return "Erro: O modelo local não foi carregado."

    # Se recebermos uma string direta ou lista de mensagens
    if isinstance(messages, str):
        prompt_text = messages
    else:
        # Extração para o formato Question/Answer
        context = ""
        user_query = ""
        for msg in messages:
            if msg["role"] == "system":
                context = msg["content"]
            elif msg["role"] == "user":
                user_query = msg["content"]
        
        prompt_text = f"Context: {context}\nQuestion: {user_query}\nAnswer:"

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.15, 
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = output_ids[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # --- PÓS-PROCESSAMENTO DE LIMPEZA ---
    
    # 1. Corta alucinações de repetição de tags
    for stop_word in ["Question:", "Answer:", "###", "Context:"]:
        if stop_word in response:
            response = response.split(stop_word)[0].strip()

    # 2. Limpa inícios de frase "quebrados" (comum em RAG)
    bad_starts = ["as well", "and ", "but ", "or ", "with ", "also "]
    for start in bad_starts:
        if response.lower().startswith(start):
            response = response[len(start):].strip().capitalize()

    # 3. Garante que a resposta termine no último sinal de pontuação (evita cortes no meio)
    # Procuramos o último ponto final, interrogação ou exclamação
    last_punc = max(response.rfind('.'), response.rfind('?'), response.rfind('!'))
    if last_punc != -1:
        response = response[:last_punc + 1]

    return response

# =========================
# CLASSE PARA CARREGAR DATASET E EMBEDDINGS
# =========================
class LLMDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = []
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def load_dataset(self):
        self.dataset = []
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.dataset.append(json.loads(line))
            print(f"✅ Dataset carregado com {len(self.dataset)} registros.")
        except Exception as e:
            print(f"❌ Erro ao ler dataset: {e}")

    def prepare_embeddings(self):
        if not self.dataset:
            return

        docs = []
        for item in self.dataset:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            content = f"Question: {question}\nAnswer: {answer}"
            
            if answer:
                metadata = {
                    "url": item.get("url", "No source"),
                    "focus": item.get("focus", "Medical Context")
                }
                docs.append(Document(page_content=content, metadata=metadata))
        
        if docs:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            print(f"✅ FAISS criado com sucesso.")
        else:
            print("❌ Erro: Documentos inválidos para FAISS.")

    def get_vectorstore(self):
        return self.vectorstore