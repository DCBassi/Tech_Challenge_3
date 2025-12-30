# graph.py

from chains import MedicalAssistantChain

def print_chat_history(chat_history):
    print("\n====== HISTÓRICO DE CHAT ======\n")
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            continue
        elif role == "user":
            print(f"🧑 Usuário: {content}")
        elif role == "assistant":
            print(f"🤖 Assistente: {content}")
    print("\n===============================\n")

def run_chat():
    chain = MedicalAssistantChain()
    chat_history = []

    print("Iniciando chat com assistente médico. Digite 'sair' para encerrar.\n")
    while True:
        try:
            user_input = input("🧑 Usuário: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando chat...")
            break

        if user_input.lower() in ["sair", "exit"]:
            print("Encerrando chat...")
            break

        response, chat_history = chain.run(user_input, chat_history)
        print(f"🤖 Assistente: {response}\n")
        print_chat_history(chat_history)

if __name__ == "__main__":
    run_chat()
