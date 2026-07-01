# Mediapipe LEO - Analise Multimodal de Linguagem Nao-Verbal

Este repositorio apresenta um prototipo para analise de linguagem nao-verbal utilizando a framework MediaPipe Holistic. O sistema processa um arquivo de video para extrair coordenadas globais de pontos de referencia (landmarks) do corpo, maos e face, dados acusticos (pitch e intensidade) e transcricao de texto alinhada por palavra, exportando os dados estruturados em arquivos CSV/JSON.

>  **Nota tecnica:** o projeto usa dois ambientes conda separados porque `mediapipe` e `whisperx` tem requisitos de dependencias incompativeis entre si.

## Estrutura do Projeto

```
mediapipe-LEO/
+-- extractors/                    # Modulos de extracao de features
|   +-- vision.py                  # MediaPipe Holistic (landmarks corporais/faciais)
|   +-- acoustics.py               # Parselmouth/Praat (pitch + intensidade)
|   +-- transcription.py           # WhisperX (transcricao alinhada por palavra)
|
+-- pipeline/                      # Orquestracao e merge dos dados
|   +-- joiner.py                  # Funcoes de merge
|   +-- extract.py                 # CLI: extracao de visao + acustica
|   +-- transcribe.py              # CLI: extracao de texto (WhisperX)
|   +-- merge.py                   # CLI: merge dos dados em dataset final
|
+-- data/                          # Dados gerados (saida)
+-- videos/                        # Videos de entrada
+-- scripts/                       # Scripts e configuracao
|   +-- ffmpeg_install.py          # Instalacao do FFmpeg standalone
|   +-- setup_environments.py      # Cria os ambientes conda automaticamente
|   +-- environment_mediapipe.yml  # Definicao do ambiente mediapipe_env
|   +-- environment_whisperx.yml   # Definicao do ambiente whisperx_env
+-- run_pipeline.py                # Roda o pipeline inteiro com um comando
+-- README.md
```

## Como instalar

### 1. Instale o Miniconda

- [Link de Download](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

Abra um terminal Anaconda (pressione a tecla Windows, digite `anaconda` e aperte `enter`).

### 2. Crie os dois ambientes automaticamente

Em vez de rodar `conda create`/`conda install` manualmente para cada ambiente, use o script abaixo. Ele cria `mediapipe_env` e `whisperx_env` a partir dos arquivos `.yml`, baixa o FFmpeg standalone, e **pula ambientes que ja existem** (seguro rodar de novo):

```bash
python scripts/setup_environments.py
```

Isso substitui todos os comandos manuais de `conda create`, `conda activate`, `conda install` e `pip install` que antes eram necessarios para montar os dois ambientes.

## Como utilizar

Coloque seus videos na pasta `videos/`. Os dados gerados serao salvos na pasta `data/`.

### Modo automatico (recomendado)

Rode o pipeline inteiro com **um unico comando**, sem precisar ativar/desativar ambientes manualmente:

```bash
python run_pipeline.py --input videos/nome_do_video.mp4
```

O script `run_pipeline.py` internamente:
1. Roda `pipeline/extract.py` dentro de `mediapipe_env` (visao + acustica)
2. Roda `pipeline/transcribe.py` dentro de `whisperx_env` (transcricao de texto)
3. Roda `pipeline/merge.py` dentro de `mediapipe_env` (gera o CSV final)

### Modo manual (passo a passo)

Se preferir o controle manual, ainda eh possivel ativar cada ambiente na mao:

```bash
conda activate mediapipe_env
python pipeline/extract.py --input videos/video.mp4
conda deactivate

conda activate whisperx_env
python pipeline/transcribe.py --input videos/video.mp4
conda deactivate

conda activate mediapipe_env
python pipeline/merge.py
```

### Argumentos Disponiveis

Possiveis configuracoes na hora de executar.
- A complexidade aumenta a precisao, entretanto tambem aumenta o tempo de processamento.
- Ajustar a confianca apenas em caso de falhas de deteccao
- Diminuir a altura e largura caso esteja muito lento o processamento (existe risco de perda de precisao)

#### Via `run_pipeline.py` (modo automatico):

```bash
python run_pipeline.py --input videos/video.mp4 \
    --mp-complexity 2 \
    --min_det 0.6 \
    --min_trk 0.6 \
    --width 1280 --height 720 \
    --draw \
    --wx-model medium \
    --lang en \
    --no-cache
```

| Argumento | Afeta | Descricao |
| :--- | :--- | :--- |
| `--input` | Todos | Caminho do video de entrada (obrigatorio) |
| `--mp-complexity` | extract.py | `0` (Lite), `1` (Normal) ou `2` (Heavy) |
| `--min_det` | extract.py | Confianca minima de deteccao (0.0-1.0) |
| `--min_trk` | extract.py | Confianca minima de rastreamento entre frames |
| `--width` / `--height` | extract.py | Reduza se o processamento estiver lento |
| `--draw` | extract.py | Abre uma janela com os landmarks desenhados |
| `--wx-model` | transcribe.py | `base`, `small`, `medium`, `large-v2` |
| `--lang` | transcribe.py | Idioma do audio (ex: `pt`, `en`) |
| `--no-cache` | merge.py | Remove arquivos intermediarios ao final |

#### Via scripts individuais (modo manual):

**pipeline/extract.py** (Visao + Acustica):
| Argumento | Tipo | Padrao | Descricao |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *(Obrigatorio)* | Caminho para o arquivo de video |
| `--complexity` | `int` | `1` | Complexidade do modelo MediaPipe: `0`, `1` ou `2` |
| `--min_det` | `float` | `0.5` | Confianca minima de deteccao |
| `--min_trk` | `float` | `0.5` | Confianca minima de rastreamento |
| `--width` | `int` | `None` | Largura de redimensionamento |
| `--height` | `int` | `None` | Altura de redimensionamento |
| `--draw` | `flag` | `False` | Mostra video com landmarks em tempo real |

**pipeline/transcribe.py** (Texto):
| Argumento | Tipo | Padrao | Descricao |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *(Obrigatorio)* | Caminho para o arquivo de video |
| `--model` | `str` | `small` | Tamanho do modelo WhisperX |
| `--lang` | `str` | `pt` | Idioma do audio |

**pipeline/merge.py** (Merge):
| Argumento | Tipo | Padrao | Descricao |
| :--- | :--- | :--- | :--- |
| `--no-cache` | `flag` | `False` | Deleta arquivos intermediarios apos a execucao |

*Atencao: apos criar os ambientes com `setup_environments.py`, nao eh necessario roda-lo de novo - apenas use `run_pipeline.py` para processar novos videos.*