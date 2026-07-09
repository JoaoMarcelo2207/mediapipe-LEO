import sys
import argparse
from pathlib import Path

# Adiciona a raiz do projeto ao path para encontrar os pacotes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractors.transcription import extrair_texto

def main():
    parser = argparse.ArgumentParser(description="Extrai transcricao de texto alinhada por palavra (WhisperX) de um video.")
    parser.add_argument("--input", type=str, required=True, help="Caminho do video MP4")
    parser.add_argument("--model", type=str, default="large-v2", help="Tamanho do modelo Whisper (ex: base, small, medium, large-v2)")
    parser.add_argument("--lang", type=str, default="pt", help="Idioma do audio (ex: pt, en)")

    args = parser.parse_args()
    video = args.input

    print(f"=== INICIANDO PIPELINE WHISPERX: {video} ===")
    print("[INFO] Extraindo texto do video...")
    try:
        extrair_texto(input_video=video, output_json="data/dados_texto.json", model_size=args.model, lang=args.lang)
    except Exception as e:
        print(f"[ERROR] Erro ao extrair texto: {e}")
    else:
        print("[INFO] Extracao de texto concluida com sucesso.")


if __name__ == "__main__":
    main()