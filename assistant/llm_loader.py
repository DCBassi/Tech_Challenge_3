from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

BASE_PATH = Path(__file__).parent.parent
MODEL_DIR = (BASE_PATH / "models" / "llama_finetuned").resolve()
MODEL_PATH = str(MODEL_DIR) if MODEL_DIR.exists() else str((BASE_PATH / "models").resolve())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        local_files_only=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
except Exception as e:
    print(f"Load error: {e}")

def safety_filter(text):
    low_text = text.lower()
    forbidden = ["prescribe", "take", "dosage", "mg/day", "javascript", "python", "recipe", "movie", "song"]
    if any(term in low_text for term in forbidden):
        return "I am sorry, but I can only assist with clinical inquiries related to the patient's specific condition."
    return text

def chat_with_model(prompt, max_new_tokens=150):
    if model is None or tokenizer is None: return "Error"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # MODO DETERMINÍSTICO: Evita alucinações e "viagens"
            repetition_penalty=1.6,  # Evita que o modelo fique preso em loops de palavras
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    
    # Limpeza de rastro e normalização de espaços
    response = re.sub(r'\s+', ' ', response) 
    stop_words = ["Question:", "###", "Context:", "<|", "SYSTEM:", "USER:", "ANSWER:"]
    for stop in stop_words:
        if stop in response:
            response = response.split(stop)[0].strip()

    return safety_filter(response)

class LLMDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = []
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    def load_dataset(self):
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): self.dataset.append(json.loads(line))
        except: pass

    def prepare_embeddings(self):
        # Reduzimos o fragmento para 400 caracteres para não confundir o modelo pequeno
        docs = [Document(page_content=f"Information: {i['answer'][:400]}", 
                         metadata={"url": i.get("url", "Internal Source")}) 
                for i in self.dataset if i.get("answer")]
        if docs:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def get_vectorstore(self):
        return self.vectorstore