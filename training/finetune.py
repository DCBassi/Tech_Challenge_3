# finetune.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import shutil
import os
from pathlib import Path

# ------------------------ # CONFIGURAÇÕES # ------------------------
MODEL_NAME = "unsloth/llama-3.2-1b" 
DATASET_PATH = "../data/processed/dataset_finetuning.jsonl"
OUTPUT_DIR = "../models/llama_finetuned"

# Garante que a pasta de destino existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------ # TOKENIZER # ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Recomendado para Llama

# ------------------------ # MODELO BASE # ------------------------
# Ajustado para carregar em CPU de forma segura se CUDA não estiver disponível
device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"🔹 Carregando modelo base em {torch_dtype}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    device_map=device_map,
    low_cpu_mem_usage=True,
)

# ------------------------ # PEFT / LoRA ------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ------------------------ # DATASET ------------------------
dataset = load_dataset("json", data_files={"train": DATASET_PATH})

def tokenize(example):
    # Criamos o prompt
    prompt = f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"
    
    # Tokenizamos
    tokenized = tokenizer(
        prompt, 
        truncation=True, 
        max_length=256, 
        padding="max_length"
    )
    
    # A CHAVE DO SUCESSO: O Trainer precisa dos 'labels' para calcular o loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# ------------------------ # TREINAMENTO ------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1, # Reduzido para evitar estouro de memória
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    save_total_limit=1,
    logging_steps=1,
    fp16=torch.cuda.is_available(), # Desativa FP16 se estiver em CPU
    use_cpu=not torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

print("🔹 Iniciando fine-tuning...")
trainer.train()

# ------------------------ # SALVANDO E MERGE ------------------------
print("🔹 Salvando adaptadores...")
temp_lora_path = "./temp_lora"
model.save_pretrained(temp_lora_path)

print("🔹 Realizando o Merge dos pesos (Método Manual Seguro)...")
# 1. Limpeza de memória RAM
import gc
del model
del trainer
gc.collect()

# 2. Carregar base SEM device_map automático para evitar offload no disco
# Usamos device_map=None para forçar o carregamento bruto na RAM
base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    device_map=None  # <--- MUDANÇA CHAVE: Desativa o mapeamento automático
)

# 3. Colocar o modelo na CPU manualmente
base_model_for_merge.to("cpu")

# 4. Carregar os adaptadores e fundir
from peft import PeftModel
merged_model = PeftModel.from_pretrained(base_model_for_merge, temp_lora_path)
merged_model = merged_model.merge_and_unload()

print(f"🔹 Salvando modelo final em {OUTPUT_DIR}...")
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Limpeza de arquivos temporários
if os.path.exists(temp_lora_path):
    shutil.rmtree(temp_lora_path)

print(f"✅ TUDO PRONTO! Modelo salvo em: {OUTPUT_DIR}")