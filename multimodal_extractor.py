import argparse

from holistic_capture_video_file import extract_holistic_landmarks
from transcription_capture_video_file import extrair_texto
from pitch_capture_video_file import extrair_acustica
from joiner import join_multimodal_data

def main():
    # 1. O orquestrador centraliza a leitura do terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Caminho do vídeo MP4")
    parser.add_argument("--draw", action="store_true", help="Desenhar landmarks do MediaPipe no vídeo")
    parser.add_argument("--complexity", type=int, default=1, help="Complexidade do modelo: 0 (lite), 1 (normal) ou 2 (slow)")
    parser.add_argument("--min_det", type=float, default=0.5, help="Confiança mínima para detecção")
    parser.add_argument("--min_trk", type=float, default=0.5, help="Confiança mínima para rastreamento")
    parser.add_argument("--width", type=int, default=None, help="Largura do vídeo para processamento(Se não informado usa original do video)")
    parser.add_argument("--height", type=int, default=None, help="Altura do vídeo para processamento(Se não informado usa original do video)")
    parser.add_argument("--model", type=str, default="large-v2", help="Tamanho do modelo Whisper (ex: base, small, medium, large-v2)")
    parser.add_argument("--lang", type=str, default="pt", help="Idioma do áudio (ex: pt, en)")
    args = parser.parse_args()

    video = args.input

    print(f"=== INICIANDO PIPELINE MULTIMODAL: {video} ===")
    
    # 2. Repassa os argumentos lidos para as funções importadas
    print("\n[1/3] Extraindo Visão (MediaPipe)...")
    extract_holistic_landmarks(
        input_path=video, 
        output_path="dados_visao.csv", 
        complexity=args.complexity,
        min_det=args.min_det,
        min_trk=args.min_trk,
        width=args.width,
        height=args.height,
        draw=args.draw
    )

    print("\n[2/3] Extraindo Texto (WhisperX)...")
    extrair_texto(input_video=video, output_json="dados_texto.json", model_size=args.model, lang=args.lang)

    print("\n[3/3] Extraindo Acústica (Parselmouth)...")
    extrair_acustica(input_video=video, output_csv="dados_acustica.csv")

    print("\n[OK] Todas as extrações finalizadas!")
    print("[INFO] Execultando integração multimodal...")
    join_multimodal_data(input_visao="dados_visao.csv",  input_acustica="dados_acustica.csv", input_texto="dados_texto.json")


# O orquestrador é o único cara que precisa disso agora
if __name__ == "__main__":
    main()