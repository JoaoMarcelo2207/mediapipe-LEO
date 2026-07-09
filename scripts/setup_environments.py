"""
Cria os ambientes conda necessarios para o pipeline (mediapipe_env e whisperx_env)
a partir dos arquivos environment_mediapipe.yml e environment_whisperx.yml,
e garante que o ffmpeg standalone esteja disponivel.

Uso:
    python scripts/setup_environments.py

E seguro rodar mais de uma vez: ambientes ja existentes sao pulados.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
ENVS_DIR = PROJECT_DIR / "environments"

ENVS = {
    "mediapipe_env": ENVS_DIR / "environment_mediapipe.yml",
    "whisperx_env": ENVS_DIR / "environment_whisperx.yml",
}


def get_existing_envs():
    result = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True, check=True
    )
    names = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0])
    return names


def setup_ffmpeg():
    """Garante que o ffmpeg standalone esteja disponivel na raiz do projeto."""
    ffmpeg_path = PROJECT_DIR / "ffmpeg.exe"
    if ffmpeg_path.exists():
        print("[OK] FFmpeg standalone ja esta disponivel.")
        return

    print("[..] FFmpeg nao encontrado. Baixando via script...")
    script = SCRIPTS_DIR / "ffmpeg_install.py"
    if not script.exists():
        print(f"[ERRO] Script nao encontrado: {script}")
        sys.exit(1)

    subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_DIR), check=True)


def main():
    try:
        existing = get_existing_envs()
    except FileNotFoundError:
        print("ERRO: comando 'conda' nao encontrado. Rode este script dentro de um")
        print("terminal Anaconda/Miniconda (Anaconda Prompt).")
        sys.exit(1)

    # 1. Garante o ffmpeg standalone
    setup_ffmpeg()

    # 2. Cria os ambientes conda
    for env_name, yml_path in ENVS.items():
        if not yml_path.exists():
            print(f"ERRO: arquivo nao encontrado: {yml_path}")
            sys.exit(1)

        if env_name in existing:
            print(f"[OK] Ambiente '{env_name}' ja existe. Pulando criacao.")
            continue

        print(f"[..] Criando ambiente '{env_name}' a partir de {yml_path.name} ...")
        subprocess.run(["conda", "env", "create", "-f", str(yml_path)], check=True)
        print(f"[OK] Ambiente '{env_name}' criado com sucesso.")

    print("\nTodos os ambientes estao prontos. Voce ja pode rodar:")
    print("    python run_pipeline.py --input videos/seu_video.mp4")


if __name__ == "__main__":
    main()
