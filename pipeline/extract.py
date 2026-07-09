import sys
import argparse
from pathlib import Path

# Adiciona a raiz do projeto ao path para encontrar os pacotes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractors.vision import extract_holistic_landmarks
from extractors.acoustics import extrair_acustica

def main():
    parser = argparse.ArgumentParser(description="Extrai dados de visao (MediaPipe) e acustica (Parselmouth) de um video.")
    parser.add_argument("--input", type=str, required=True, help="Caminho do video MP4")
    parser.add_argument("--draw", action="store_true", help="Desenhar landmarks do MediaPipe no video")
    parser.add_argument("--complexity", type=int, default=1, help="Complexidade do modelo: 0 (lite), 1 (normal) ou 2 (slow)")
    parser.add_argument("--min_det", type=float, default=0.5, help="Confianca minima para deteccao")
    parser.add_argument("--min_trk", type=float, default=0.5, help="Confianca minima para rastreamento")
    parser.add_argument("--width", type=int, default=None, help="Largura do video para processamento")
    parser.add_argument("--height", type=int, default=None, help="Altura do video para processamento")

    args = parser.parse_args()
    video = args.input

    print(f"=== INICIANDO PIPELINE MEDIAPIPE + PARSELMOUTH: {video} ===")
    
    print("\n[1/2] Extraindo Visao (MediaPipe)...")
    try:
        extract_holistic_landmarks(
            input_path=video, 
            output_path="data/dados_visao.csv", 
            complexity=args.complexity,
            min_det=args.min_det,
            min_trk=args.min_trk,
            width=args.width,
            height=args.height,
            draw=args.draw
        )
    except Exception as e:
        print(f"[ERROR] Erro ao extrair visao: {e}")
    else:
        print("[INFO] Extracao de visao concluida com sucesso.")
        print("\n[2/2] Extraindo Acustica (Parselmouth)...")
        try:
            extrair_acustica(input_video=video, output_path="data/dados_acustica.csv")
        except Exception as e:
            print(f"[ERROR] Erro ao extrair acustica: {e}")
        else:
            print("[INFO] Extracao de acustica concluida com sucesso.")


if __name__ == "__main__":
    main()