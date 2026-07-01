import os
import json
import numpy as np
import pandas as pd


def join_multimodal_data(arq_visao, arq_acustica, arq_texto, saida_csv="data/dataset_multimodal.csv"):
    print("1. Carregando os dados brutos...")
    
    try:
        df_visao = pd.read_csv(arq_visao, index_col=False)
        df_acustica = pd.read_csv(arq_acustica, index_col=False)

        df_visao['timestamp_ms'] = df_visao['timestamp_ms'].astype(float)
        df_acustica['timestamp_ms'] = df_acustica['timestamp_ms'].astype(float)

        df_visao = df_visao.sort_values('timestamp_ms')
        df_acustica = df_acustica.sort_values('timestamp_ms')
    except Exception as e:
        print(f"[ERRO FATAL] O Pandas falhou na leitura: {e}")
        return

    print("2. Juntando Video e Audio...")
    df_final = pd.merge_asof(
        df_visao, 
        df_acustica, 
        on='timestamp_ms', 
        direction='nearest'
    )
    df_final = df_final.copy()

    print("3. Inserindo as palavras do WhisperX...")
    df_final['palavra'] = "SILENCIO" 
    
    try:
        with open(arq_texto, 'r', encoding='utf-8') as f:
            dados_texto = json.load(f)

        lista_palavras = []
        if isinstance(dados_texto, list):
            for item in dados_texto:
                if 'words' in item:  
                    lista_palavras.extend(item['words'])
                elif 'word' in item: 
                    lista_palavras.append(item)
        elif isinstance(dados_texto, dict):
            lista_palavras = dados_texto.get('word_segments', [])
            if not lista_palavras and 'segments' in dados_texto:
                for seg in dados_texto['segments']:
                    if 'words' in seg:
                        lista_palavras.extend(seg['words'])

        inseridas = 0
        for p in lista_palavras:
            if 'start' in p and 'end' in p:
                inicio_ms = p['start'] * 1000
                fim_ms = p['end'] * 1000
                
                mascara = (df_final['timestamp_ms'] >= inicio_ms) & (df_final['timestamp_ms'] <= fim_ms)
                if mascara.any():
                    df_final.loc[mascara, 'palavra'] = p['word']
                    inseridas += 1

        print(f"-> Palavras encaixadas no CSV com sucesso: {inseridas}")

    except Exception as e:
        print(f"[ERRO] Falha ao processar o JSON: {e}")

    print("4. Salvando dataset final...")
    cols_numericas = df_final.select_dtypes(include=[np.number]).columns
    df_final[cols_numericas] = df_final[cols_numericas].fillna(0.0)

    df_final.to_csv(saida_csv, index=False)
    print("Concluido!")


def cleanup_intermediate_files():
    """Remove os arquivos intermediarios gerados pelos extractors."""
    for arquivo in ["data/dados_visao.csv", "data/dados_acustica.csv", "data/dados_texto.json"]:
        if os.path.exists(arquivo):
            os.remove(arquivo)
    print("[INFO] Arquivos temporarios removidos com sucesso.")