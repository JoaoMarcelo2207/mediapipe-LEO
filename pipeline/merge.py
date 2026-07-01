import sys
import argparse
from pathlib import Path

# Adiciona a raiz do projeto ao path para encontrar os pacotes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.joiner import join_multimodal_data, cleanup_intermediate_files

def main():
    parser = argparse.ArgumentParser(description="Junta os dados de visao, acustica e texto em um dataset multimodal unico.")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Deleta os arquivos intermediarios apos a execucao")

    args = parser.parse_args()

    print("[INFO] Executando integracao multimodal...")
    try:
        join_multimodal_data(
            arq_visao="data/dados_visao.csv", 
            arq_acustica="data/dados_acustica.csv", 
            arq_texto="data/dados_texto.json"
        )
    except Exception as e:
        print(f"[ERROR] Erro ao integrar dados multimodais: {e}")
    else:
        print("[INFO] Integracao multimodal concluida com sucesso.")
        if args.no_cache:
            try:
                cleanup_intermediate_files()
            except Exception as e:
                print(f"[ERROR] Falha ao remover arquivos temporarios: {e}")


if __name__ == "__main__":
    main()