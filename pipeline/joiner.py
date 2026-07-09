import os
import json
import numpy as np
import pandas as pd


def _interpolar_para_grade(df, grade_ms, colunas_nearest=None):
    """
    Interpola um DataFrame para uma grade temporal regular usando interpolacao linear.

    Passos:
      1. Combina os timestamps originais do DataFrame com a grade alvo.
      2. Reindexar faz com que os pontos novos tenham NaN.
      3. Interpolacao linear (method='index') preenche os NaN com base na
         distancia real entre os timestamps, nao pela posicao no array.
      4. bfill/ffill cobre as bordas (onde nao ha dois pontos para interpolar).
      5. Filtra de volta apenas os pontos que pertencem a grade.

    Args:
        df: DataFrame com coluna 'timestamp_ms' e dados numericos.
        grade_ms: Array numpy com os timestamps alvo (em ms).
        colunas_nearest: Lista de colunas para usar interpolacao 'nearest'
                         em vez de linear (ex: coluna 'frame').

    Returns:
        DataFrame interpolado com exatamente os timestamps da grade.
    """
    if colunas_nearest is None:
        colunas_nearest = []

    df_indexed = df.set_index('timestamp_ms')

    # Combina os timestamps originais com a grade para preservar os valores conhecidos
    indice_combinado = df_indexed.index.union(pd.Index(grade_ms)).sort_values()
    indice_combinado = indice_combinado.drop_duplicates()

    # Reindexar: valores originais ficam, novos pontos da grade viram NaN
    df_expandido = df_indexed.reindex(indice_combinado)

    # Separar colunas para interpolacao linear vs nearest
    cols_linear = [c for c in df_expandido.columns if c not in colunas_nearest]
    cols_nn = [c for c in df_expandido.columns if c in colunas_nearest]

    # Interpolacao linear (usando o indice numerico para espacamento correto)
    if cols_linear:
        df_expandido[cols_linear] = df_expandido[cols_linear].interpolate(method='index')

    # Interpolacao nearest para colunas discretas (ex: numero do frame)
    if cols_nn:
        df_expandido[cols_nn] = df_expandido[cols_nn].interpolate(method='nearest')

    # Preencher as bordas (timestamps antes do primeiro dado ou depois do ultimo)
    # bfill: preenche o inicio com o primeiro valor disponivel
    # ffill: preenche o final com o ultimo valor disponivel
    df_expandido = df_expandido.bfill().ffill()

    # Filtrar apenas os pontos da grade regular
    df_resultado = df_expandido.loc[df_expandido.index.isin(grade_ms)].copy()
    df_resultado.index.name = 'timestamp_ms'
    df_resultado = df_resultado.reset_index()

    return df_resultado


def join_multimodal_data(arq_visao, arq_acustica, arq_texto, saida_csv="data/dataset_multimodal.csv"):
    print("1. Carregando os dados brutos...")

    try:
        df_visao = pd.read_csv(arq_visao, index_col=False)
        df_acustica = pd.read_csv(arq_acustica, index_col=False)

        df_visao['timestamp_ms'] = df_visao['timestamp_ms'].astype(float)
        df_acustica['timestamp_ms'] = df_acustica['timestamp_ms'].astype(float)

        df_visao = df_visao.sort_values('timestamp_ms').reset_index(drop=True)
        df_acustica = df_acustica.sort_values('timestamp_ms').reset_index(drop=True)
    except Exception as e:
        print(f"[ERRO FATAL] O Pandas falhou na leitura: {e}")
        return

    # --- Grade temporal unificada a 10ms ---
    print("2. Criando grade temporal unificada (passo = 10ms)...")

    # Pega o menor e o maior timestamp dentre TODAS as fontes
    t_min = min(df_visao['timestamp_ms'].min(), df_acustica['timestamp_ms'].min())
    t_max = max(df_visao['timestamp_ms'].max(), df_acustica['timestamp_ms'].max())

    # Arredondar para multiplos de 10: t_min para baixo, t_max para cima
    t_min_grid = np.floor(t_min / 10) * 10
    t_max_grid = np.ceil(t_max / 10) * 10

    grade_ms = np.arange(t_min_grid, t_max_grid + 10, 10)
    print(f"   Grade: {t_min_grid:.0f}ms -> {t_max_grid:.0f}ms ({len(grade_ms)} pontos)")

    # --- Interpolar Visao (de ~33ms para 10ms) ---
    print("3. Interpolando dados de visao para grade de 10ms...")
    n_original_visao = len(df_visao)
    df_visao_interp = _interpolar_para_grade(df_visao, grade_ms, colunas_nearest=['frame'])

    # Coluna 'frame' deve ser inteira (nearest pode gerar float)
    if 'frame' in df_visao_interp.columns:
        df_visao_interp['frame'] = df_visao_interp['frame'].astype(int)

    print(f"   Visao: {n_original_visao} -> {len(df_visao_interp)} pontos (interpolados)")

    # --- Regularizar Acustica na mesma grade ---
    print("4. Regularizando dados acusticos na mesma grade...")
    n_original_acustica = len(df_acustica)
    df_acustica_interp = _interpolar_para_grade(df_acustica, grade_ms)
    print(f"   Acustica: {n_original_acustica} -> {len(df_acustica_interp)} pontos")

    # --- Juntar tudo ---
    print("5. Juntando visao + acustica...")

    # Ambos tem exatamente os mesmos timestamps agora; basta concatenar colunas
    df_acustica_sem_ts = df_acustica_interp.drop(columns=['timestamp_ms'])
    df_final = pd.concat([df_visao_interp, df_acustica_sem_ts], axis=1)

    # --- Inserir Texto (WhisperX) ---
    print("6. Inserindo as palavras do WhisperX...")
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

    # --- Salvar ---
    print("7. Salvando dataset final...")
    cols_numericas = df_final.select_dtypes(include=[np.number]).columns
    df_final[cols_numericas] = df_final[cols_numericas].fillna(0.0)

    df_final.to_csv(saida_csv, index=False)
    print(f"[OK] Concluido! Dataset final: {len(df_final)} linhas, {len(df_final.columns)} colunas")
    print(f"     Resolucao temporal: 10ms | Intervalo: {grade_ms[0]:.0f}ms -> {grade_ms[-1]:.0f}ms")


def cleanup_intermediate_files():
    """Remove os arquivos intermediarios gerados pelos extractors."""
    for arquivo in ["data/dados_visao.csv", "data/dados_acustica.csv", "data/dados_texto.json"]:
        if os.path.exists(arquivo):
            os.remove(arquivo)
    print("[INFO] Arquivos temporarios removidos com sucesso.")
