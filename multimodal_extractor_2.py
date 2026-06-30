import argparse
from html import parser
from joiner import join_multimodal_data
from transcription_capture_video_file import extrair_texto


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Caminho do vídeo MP4")
    parser.add_argument("--model", type=str, default="small", help="Tamanho do modelo Whisper (ex: base, small, medium, large-v2)")
    parser.add_argument("--lang", type=str, default="pt", help="Idioma do áudio (ex: pt, en)")

    args = parser.parse_args()
    video = args.input

    print(f"=== INICIANDO PIPELINE WHISPERX: {video} ===")
    print("[INFO] Extraindo texto do vídeo...")
    try:
        extrair_texto(input_video=video, output_json="dados_texto.json", model_size=args.model, lang=args.lang)
    except Exception as e:
        print(f"[ERROR] Erro ao extrair texto: {e}")
    else:
        print("[INFO] Extração de texto concluída com sucesso.")
    

if __name__ == "__main__":
    main()