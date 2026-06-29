import pandas as pd

def join_multimodal_data(arq_visao, arq_acustica, arq_texto, saida_csv="dataset_multimodal.csv"):
    
    print("1. Carregando os três arquivos...")
    try:
        df_visao = pd.read_csv(arq_visao)       # Seu CSV gigante do MediaPipe
        df_acustica = pd.read_csv(arq_acustica) # Seu CSV do Parselmouth
        df_texto = pd.read_json(arq_texto)      # Seu JSON do WhisperX
    except Exception as e:
        print(f"Erro ao carregar os arquivos: {e}")
        return

    # Para juntar dados de tempo, o Pandas exige que as tabelas estejam em ordem crescente
    df_visao = df_visao.sort_values('timestamp_ms')
    df_acustica = df_acustica.sort_values('timestamp_ms')

    print("2. Juntando Vídeo e Áudio...")
    # O "merge_asof" é a ferramenta que junta as coisas pelo tempo "mais próximo".
    # Ele pega a linha do MediaPipe e cola nela o áudio do exato milissegundo correspondente.
    df_final = pd.merge_asof(
        df_visao, 
        df_acustica, 
        on='timestamp_ms', 
        direction='nearest'
    )

    print("3. Colocando as palavras no tempo certo...")
    # Cria uma coluna vazia chamada 'palavra'
    df_final['palavra'] = "" 
    
    # Preenche a palavra nos frames em que ela foi falada
    for _, row in df_texto.iterrows():
        palavra = row['word']
        inicio_ms = row['start'] * 1000
        fim_ms = row['end'] * 1000
        
        # Filtra os frames que aconteceram dentro do tempo da palavra
        mascara_tempo = (df_final['timestamp_ms'] >= inicio_ms) & (df_final['timestamp_ms'] <= fim_ms)
        df_final.loc[mascara_tempo, 'palavra'] = palavra

    print("4. Salvando o arquivo final...")
    # Troca possíveis erros de cálculo nulos por 0.0 para não sujar sua tabela
    df_final = df_final.fillna(0.0) 
    
    df_final.to_csv(saida_csv, index=False)
    print(f"Sucesso! Dataset criado com {len(df_final.columns)} colunas.")