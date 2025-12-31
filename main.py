from assistant.llm_loader import LLMDataLoader
from assistant.chains import create_qa_chain
from assistant.logger import log_interaction
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent
    dataset_path = BASE_DIR / "data" / "processed" / "dataset_finetuning.jsonl"
    
    # 1. Inicializa o carregador e o banco de dados vetorial
    loader = LLMDataLoader(dataset_path)
    loader.load_dataset()
    loader.prepare_embeddings()
    vectorstore = loader.get_vectorstore()
    
    print("\n" + "="*40)
    print("🏥 MEDICAL QA SYSTEM - Llama-3.2 Fine-tuned")
    print("="*40)

    # REQUISITO: Consulta em base estruturada e Contextualização
    # Simulamos a entrada do ID do paciente para buscar no prontuário (base estruturada)
    patient_id = input("\n🆔 Digite o ID do paciente para consulta (ex: 123 ou 456): ").strip()
    
    # 2. Cria a chain passando o ID do paciente
    # A chain agora integra LLM + RAG + Prontuário
    qa_chain = create_qa_chain(vectorstore, patient_id=patient_id)

    print(f"\n✅ Atendimento iniciado para o Paciente ID: {patient_id}")

    while True:
        try:
            question = input("\n🧑 Digite sua pergunta clínica: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

        if question.lower() in ["exit", "quit"]:
            print("Encerrando...")
            break
        
        if not question:
            continue

        # 3. Executa a Chain com LangChain Pipeline
        result = qa_chain(question)
        
        answer = result.get("result", "Sem resposta disponível.")
        docs = result.get("source_documents", [])
        patient_context = result.get("patient_context", "Sem dados")
        strategy = result.get("strategy", "unknown")
        source_urls = [d.metadata.get("url", "Sem fonte") for d in docs] if docs else []

        log_interaction(
            patient_id=patient_id,
            question=question,
            answer=answer,
            strategy=strategy,
            sources=source_urls
        )
        print("Log registrado com sucesso")

        # 4. Exibe a resposta formatada
        print("\n📝 Contexto do Paciente Identificado:")
        print(f"   > {patient_context}")
        
        print("\n🤖 Resposta da LLM:")
        print("-" * 20)
        print(answer)
        print("-" * 20)

        # 5. Exibe fontes
        if docs:
            print("\n📚 Fontes Consultadas:")
            unique_sources = set([d.metadata.get("url", "Sem fonte") for d in docs])
            for s in unique_sources:
                print(f"🔗 {s}")

if __name__ == "__main__":
    main()