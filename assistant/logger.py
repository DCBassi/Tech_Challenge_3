import json
import os
from datetime import datetime
from pathlib import Path

def log_interaction(patient_id, question, answer, strategy, sources, log_file="interaction_logs.jsonl"):
    """
    Registra a interação do usuário com o sistema.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    LOG_PATH = BASE_DIR / "logs"
    
    # Cria a pasta logs se não existir
    LOG_PATH.mkdir(exist_ok=True)
    full_path = LOG_PATH / log_file

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "question": question,
        "answer": answer,
        "strategy_used": strategy,  # CSV ou RAG
        "sources": sources
    }

    # Salva no arquivo 
    with open(full_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")