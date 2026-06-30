import argparse

from holistic_capture_video_file import extract_holistic_landmarks
from pitch_capture_video_file import extrair_acustica

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

    args = parser.parse_args()

    video = args.input

    print(f"=== INICIANDO PIPELINE MEDIAPIPE + PARSELMOUTH: {video} ===")
    
    # 2. Repassa os argumentos lidos para as funções importadas
    print("\n[1/2] Extraindo Visão (MediaPipe)...")
    try:
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
    except Exception as e:
        print(f"[ERROR] Erro ao extrair visão: {e}")
    else:
        print("[INFO] Extração de visão concluída com sucesso.")
        print("\n[2/2] Extraindo Acústica (Parselmouth)...")
        try:
            extrair_acustica(input_video=video, output_path="dados_acustica.csv")
        except Exception as e:
            print(f"[ERROR] Erro ao extrair acústica: {e}")
        else:
            print("[INFO] Extração de acústica concluída com sucesso.")
  
    

if __name__ == "__main__":
    main()
