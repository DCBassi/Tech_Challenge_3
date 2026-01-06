import os
import json
import torch
import shutil
import gc
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel

# =========================
# CHECK CUDA
# =========================
assert torch.cuda.is_available(), "❌ CUDA NÃO DISPONÍVEL! Verifique seu ambiente."

print("✅ CUDA disponível")
print("🟢 GPU:", torch.cuda.get_device_name(0))

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (BASE_DIR / "data" / "processed" / "dataset_finetuning.jsonl").resolve()
OUTPUT_DIR = (BASE_DIR / "models" / "llama_finetuned").resolve()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# TOKENIZER
# =========================
MODEL_NAME = "unsloth/llama-3.2-1b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# MODEL (GPU)
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},  # FORÇA GPU
    low_cpu_mem_usage=True,
)

# =========================
# LoRA
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =========================
# DATASET
# =========================
dataset = load_dataset(
    "json",
    data_files={"train": str(DATASET_PATH).replace("\\", "/")}
)

def tokenize(example):
    prompt = (
        f"Question: {example['question']}\n"
        f"Answer: {example['answer']}{tokenizer.eos_token}"
    )
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=128,  # SEGURO PARA 6GB
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_total_limit=1,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

print("🚀 Iniciando fine-tuning na GPU...")
trainer.train()

# =========================
# SAVE LoRA
# =========================
print("💾 Salvando adaptadores LoRA...")
temp_lora_path = str((BASE_DIR / "temp_lora").resolve())
model.save_pretrained(temp_lora_path)

# =========================
# CLEAN GPU
# =========================
del trainer
del model
gc.collect()
torch.cuda.empty_cache()

# =========================
# MERGE (CPU)
# =========================
print("🔗 Merge dos pesos LoRA (CPU)...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
base_model.to("cpu")

merged_model = PeftModel.from_pretrained(base_model, temp_lora_path)
merged_model = merged_model.merge_and_unload()

# =========================
# SAVE FINAL MODEL
# =========================
merged_model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

shutil.rmtree(temp_lora_path, ignore_errors=True)

print(f"✅ FINE-TUNING CONCLUÍDO COM SUCESSO!")
print(f"📦 Modelo salvo em: {OUTPUT_DIR}")
