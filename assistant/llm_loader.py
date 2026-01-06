from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import json
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
    # Verificacao de Seguranca e Escopo na Saida
    # Se a LLM tentar responder algo fora de escopo, o filtro bloqueia.
    low_text = text.lower()
    
    # Lista de bloqueio final
    forbidden = ["prescribe", "take", "dosage", "mg/day", "javascript", "python", "clock", "time is", "the date is","recipe", "code", "movie", "song", "game","book","shopping","travel","joke"]
    
    if any(term in low_text for term in forbidden):
        return "OUT OF SCOPE / SAFETY ALERT: This request cannot be fulfilled as it is outside the clinical support boundaries."
    
    return text

def chat_with_model(prompt, max_new_tokens=450):
    if model is None or tokenizer is None: return "Error"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

    # Limpeza original de rastro
    stop_words = ["Question:", "###", "Context:", "<|"]
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
        # Explainability: URLs nos metadados
        docs = [Document(page_content=f"Information: {i['answer']}", metadata={"url": i.get("url", "Internal Source")}) for i in self.dataset if i.get("answer")]
        if docs:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def get_vectorstore(self):
        return self.vectorstore