Este repositório apresenta um protótipo para análise de linguagem não-verbal utilizando a framework MediaPipe Holistic. O sistema processa um arquivo de vídeo para extrair coordenadas globais de pontos de referência (landmarks) do corpo, mãos e face, exportando os dados estruturados em um arquivo CSV.

# Como instalar

## Usando o anaconda para criar um ambiente e instalar os pacotes

Instale miniconda (a versão menor do anaconda apenas com termina)

- [Link de Download](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

Abra um terminal miniconda

- Pressione a tecla Windows no seu teclado e digite 'anaconda' e aperte ```enter```

Com o terminal aberto siga as instruções abaixo

## Instalando pacotes Conda
Atualize os pacotes conda
```bash
conda update --all
```
Crie um ambiente (aqui um ambiente chamado mediapipe_holistic_env será usado para instalar os pacotes)
```bash
conda create --name mediapipe_holistic_env python=3.12.13 pip cmake ipython jupyter
```
Ative o ambiente
```bash
conda activate mediapipe_holistic_env
```
Instale os pacotes conda
```bash
conda install -c conda-forge opencv=4.10.0
```
Instale os pacotes pip
```bash
pip install mediapipe==0.10.14
```
# Como utilizar
Após instalar os pacotes mude para o diretório onde você clonou o repositorio
```bash
cd diretorio_do_repositorio
```
Execute e passe o diretorio do video como argumento (o ambiente deve estar ativo)
se o video estiver na mesma pasta que o arquivo .py, apenas passe o nome_do_arquivo_do_video.mp4
```bash
python holistic_landmarks_video_file.py --input nome_do_arquivo_do_video.mp4
```
### Argumentos Disponíveis
Possiveis configurações na hora de executar.
- A complexidade aumenta a precisão, entretanto tambem aumenta o tempo de processamento.
- Ajustar a confiança apenas em caso de falhas de detectação
- Diminuir a altura e largura caso esteja muito lento o processamento (existe risco de perda de precisão)

| Argumento | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *(Obrigatório)* | Caminho para o arquivo de vídeo que será processado. |
| `--output` | `str` | `holistic_landmarks...csv` | Nome do arquivo CSV onde as coordenadas serão salvas. |
| `--complexity` | `int` | `1` | Complexidade do modelo MediaPipe: `0` (Lite), `1` (Normal) ou `2` (Heavy). |
| `--min_det` | `float` | `0.5` | Valor mínimo de confiança (0.0 a 1.0) para que a detecção seja considerada bem-sucedida. |
| `--min_trk` | `float` | `0.5` | Valor mínimo de confiança para o rastreamento dos landmarks entre os frames. |
| `--width` | `int` | `640` | Largura de redimensionamento do vídeo para o processamento. |
| `--height` | `int` | `480` | Altura de redimensionamento do vídeo para o processamento. |
| `--draw` | `flag` | `False` | Se presente, abre uma janela mostrando o vídeo com os landmarks desenhados em tempo real. |
| `--help` | `flag` | `False` | Retorna esta tabela no terminal. |




