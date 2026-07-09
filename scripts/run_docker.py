"""
Orquestrador do pipeline via Docker.

Faz o mesmo que run_pipeline.py, mas usando containers Docker em vez de ambientes Conda.
Cada etapa roda em sua propria imagem (mediapipe-leo-mp ou mediapipe-leo-wx), com as
pastas videos/ e data/ montadas como volumes para compartilhar I/O entre os containers.

Uso basico:
    python scripts/run_docker.py --input videos/video.mp4

Uso com opcoes avancadas:
    python scripts/run_docker.py --input videos/video.mp4 --mp-complexity 2 --wx-model medium --lang en --no-cache

Uso com GPU (requer NVIDIA Container Toolkit):
    python scripts/run_docker.py --input videos/video.mp4 --gpu

Na primeira execucao, as imagens Docker serao construidas automaticamente.
Para reconstruir manualmente:
    docker build -f environments/Dockerfile.mediapipe -t mediapipe-leo-mp .
    docker build -f environments/Dockerfile.whisperx  -t mediapipe-leo-wx .
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_MP = "mediapipe-leo-mp"
IMAGE_WX = "mediapipe-leo-wx"


def image_exists(image_name: str) -> bool:
    """Verifica se uma imagem Docker ja existe localmente."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
    )
    return result.returncode == 0


def build_image(dockerfile: str, image_name: str) -> None:
    """Constroi uma imagem Docker a partir de um Dockerfile."""
    print(f"\n[BUILD] Construindo imagem '{image_name}' a partir de {dockerfile}...")
    cmd = [
        "docker", "build",
        "-f", str(PROJECT_DIR / dockerfile),
        "-t", image_name,
        str(PROJECT_DIR),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERRO] Falha ao construir imagem '{image_name}'.")
        sys.exit(result.returncode)
    print(f"[OK] Imagem '{image_name}' construida com sucesso.")


def ensure_images(rebuild: bool = False) -> None:
    """Garante que ambas as imagens Docker existem, construindo se necessario."""
    images = {
        IMAGE_MP: "environments/Dockerfile.mediapipe",
        IMAGE_WX: "environments/Dockerfile.whisperx",
    }
    for name, dockerfile in images.items():
        if rebuild or not image_exists(name):
            build_image(dockerfile, name)
        else:
            print(f"[OK] Imagem '{name}' ja existe. Use --rebuild para reconstruir.")


def run_in_docker(image_name: str, script: str, args: list[str], gpu: bool = False) -> None:
    """Executa um script Python dentro de um container Docker."""
    volumes = [
        "-v", f"{PROJECT_DIR / 'videos'}:/app/videos",
        "-v", f"{PROJECT_DIR / 'data'}:/app/data",
    ]

    gpu_flags = ["--gpus", "all"] if gpu else []

    cmd = (
        ["docker", "run", "--rm"]
        + gpu_flags
        + volumes
        + [image_name, script]
        + args
    )

    print(f"\n>>> [Docker:{image_name}] python {script} {' '.join(args)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERRO] '{script}' falhou no container '{image_name}' (codigo {result.returncode}).")
        print("Pipeline interrompido.")
        sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline multimodal completo (via Docker)")
    parser.add_argument("--input", required=True, help="Caminho do arquivo de video")

    # Opcoes do extract.py (MediaPipe + Parselmouth)
    parser.add_argument("--mp-complexity", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--min_det", type=float, default=None)
    parser.add_argument("--min_trk", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)

    # Opcoes do transcribe.py (WhisperX)
    parser.add_argument(
        "--wx-model", default=None, choices=["base", "small", "medium", "large-v2"],
        help="Tamanho do modelo WhisperX",
    )
    parser.add_argument("--lang", default=None, help="Idioma do audio (ex: pt, en)")

    # Opcoes do merge.py
    parser.add_argument("--no-cache", action="store_true", help="Remove arquivos intermediarios ao final")

    # Docker-specific
    parser.add_argument("--gpu", action="store_true", help="Habilita GPU NVIDIA (requer NVIDIA Container Toolkit)")
    parser.add_argument("--rebuild", action="store_true", help="Forca reconstrucao das imagens Docker")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Verificar se Docker esta disponivel
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("[ERRO] Docker nao encontrado. Instale o Docker Desktop:")
        print("       https://www.docker.com/products/docker-desktop/")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("[ERRO] Docker nao esta rodando. Inicie o Docker Desktop e tente novamente.")
        sys.exit(1)

    # Construir imagens se necessario
    ensure_images(rebuild=args.rebuild)

    # --- Flag --draw nao funciona no Docker (sem display grafico) ---
    print("\n[INFO] Nota: a flag --draw nao esta disponivel no modo Docker (sem interface grafica).")

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

    run_in_docker(IMAGE_MP, "pipeline/extract.py", extract_args, gpu=False)

    # --- Script 2: transcricao (WhisperX) ---
    transcribe_args = ["--input", args.input]
    if args.wx_model is not None:
        transcribe_args += ["--model", args.wx_model]
    if args.lang is not None:
        transcribe_args += ["--lang", args.lang]

    run_in_docker(IMAGE_WX, "pipeline/transcribe.py", transcribe_args, gpu=args.gpu)

    # --- Script 3: merge ---
    merge_args = ["--no-cache"] if args.no_cache else []
    run_in_docker(IMAGE_MP, "pipeline/merge.py", merge_args, gpu=False)

    print("\n[OK] Pipeline concluido com sucesso!")


if __name__ == "__main__":
    main()
