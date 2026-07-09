import json
import whisperx
import torch
from pathlib import Path

def extrair_texto(input_video, output_json, model_size="small", lang="pt"):
    
    input_path = Path(input_video)
    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_video}")
        return

    # Autodetecta se tem GPU (CUDA) disponível para acelerar o processo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Se for CPU, precisa usar float32 ou int8. Se for GPU, float16 é mais rápido.
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16 if device == "cuda" else 4

    print(f"[*] Iniciando WhisperX usando dispositivo: {device.upper()}")
    print(f"[*] Carregando modelo base ({model_size})...")
    
    # 1. Carrega o modelo de transcrição
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    
    # Carrega o áudio extraindo direto do MP4
    print(f"[*] Lendo áudio de {input_video}...")
    audio = whisperx.load_audio(str(input_path))
    
    # 2. Faz a transcrição bruta (cria blocos de texto)
    print("[*] Transcrevendo áudio...")
    result = model.transcribe(audio, batch_size=batch_size, language=lang)
    
    # 3. Carrega o modelo de Alinhamento (É aqui que a mágica do timestamp por palavra acontece)
    print("[*] Carregando modelo de alinhamento fonético...")
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
    
    # 4. Alinha o texto com o áudio
    print("[*] Alinhando palavras com os milissegundos...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # 5. Limpeza e formatação dos dados para o Pandas
    # O WhisperX retorna uma estrutura complexa. Vamos "achatar" (flatten) 
    # para uma lista simples de dicionários: [{'word': 'oi', 'start': 1.2, 'end': 1.5}]
    flat_words = []
    
    for segment in result["segments"]:
        for word in segment.get("words", []):
            # As vezes o modelo falha em alinhar uma palavra de respiro/pausa, 
            # então filtramos apenas palavras que ganharam timestamp.
            if 'start' in word and 'end' in word:
                flat_words.append({
                    "word": word["word"].strip().lower(), # Limpa a string
                    "start": word["start"],
                    "end": word["end"],
                    "score": round(word.get("score", 0.0), 3) # Grau de confiança da IA
                })

    # Salva o arquivo JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(flat_words, f, ensure_ascii=False, indent=4)
        
    print(f"\n[OK] Concluído! {len(flat_words)} palavras extraídas e salvas em {output_json}")