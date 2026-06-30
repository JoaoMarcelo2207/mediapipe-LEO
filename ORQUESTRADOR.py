
from joiner import join_multimodal_data

def main():
    print("[INFO] Execultando integração multimodal...")
    try:
        join_multimodal_data(arq_visao="dados_visao.csv", arq_acustica="dados_acustica.csv", arq_texto="dados_texto.json")
    except Exception as e:
        print(f"[ERROR] Erro ao integrar dados multimodais: {e}")
    else:
        print("[INFO] Integração multimodal concluída com sucesso.")

if __name__ == "__main__":
    main()