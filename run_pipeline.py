"""
Orquestrador do pipeline completo.

Roda, em sequencia e no ambiente conda correto para cada um:
  1. pipeline/extract.py       (ambiente: mediapipe_env)  - visao + acustica
  2. pipeline/transcribe.py    (ambiente: whisperx_env)   - transcricao de texto
  3. pipeline/merge.py         (ambiente: mediapipe_env)   - merge dos dados

Sem precisar ativar/desativar ambientes manualmente: usa 'conda run' por baixo.

Uso basico:
    python run_pipeline.py --input videos/video.mp4

Uso com opcoes avancadas:
    python run_pipeline.py --input videos/video.mp4 --mp-complexity 2 --wx-model medium --lang en --no-cache

Requisito: os ambientes 'mediapipe_env' e 'whisperx_env' ja devem existir.
Se ainda nao existirem, rode antes: python scripts/setup_environments.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def run_in_env(env_name: str, script: str, args: list[str]) -> None:
    cmd = ["conda", "run", "-n", env_name, "--no-capture-output", "python", script] + args
    print(f"\n>>> [{env_name}] python {script} {' '.join(args)}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"\n[ERRO] '{script}' falhou no ambiente '{env_name}' (codigo {result.returncode}).")
        print("Pipeline interrompido.")
        sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline multimodal completo")
    parser.add_argument("--input", required=True, help="Caminho do arquivo de video")

    # Opcoes do extract.py (MediaPipe + Parselmouth)
    parser.add_argument("--mp-complexity", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--min_det", type=float, default=None)
    parser.add_argument("--min_trk", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--draw", action="store_true")

    # Opcoes do transcribe.py (WhisperX)
    parser.add_argument(
        "--wx-model", default=None, choices=["base", "small", "medium", "large-v2"],
        help="Tamanho do modelo WhisperX",
    )
    parser.add_argument("--lang", default=None, help="Idioma do audio (ex: pt, en)")

    # Opcoes do merge.py
    parser.add_argument("--no-cache", action="store_true", help="Remove arquivos intermediarios ao final")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # --- Script 1: extracao de visao + acustica (MediaPipe + Parselmouth) ---
    extract_args = ["--input", args.input]
    if args.mp_complexity is not None:
        extract_args += ["--complexity", str(args.mp_complexity)]
    if args.min_det is not None:
        extract_args += ["--min_det", str(args.min_det)]
    if args.min_trk is not None:
        extract_args += ["--min_trk", str(args.min_trk)]
    if args.width is not None:
        extract_args += ["--width", str(args.width)]
    if args.height is not None:
        extract_args += ["--height", str(args.height)]
    if args.draw:
        extract_args += ["--draw"]

    run_in_env("mediapipe_env", "pipeline/extract.py", extract_args)

    # --- Script 2: transcricao (WhisperX) ---
    transcribe_args = ["--input", args.input]
    if args.wx_model is not None:
        transcribe_args += ["--model", args.wx_model]
    if args.lang is not None:
        transcribe_args += ["--lang", args.lang]

    run_in_env("whisperx_env", "pipeline/transcribe.py", transcribe_args)

    # --- Script 3: junta tudo em um CSV unico ---
    merge_args = ["--no-cache"] if args.no_cache else []
    run_in_env("mediapipe_env", "pipeline/merge.py", merge_args)

    print("\n[OK] Pipeline concluido com sucesso!")


if __name__ == "__main__":
    main()