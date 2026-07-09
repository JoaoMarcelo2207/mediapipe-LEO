import os
import csv
import tempfile
import subprocess
import numpy as np
import parselmouth
from pathlib import Path

def extract_audio_from_video(video_path, wav_path):
    """Extrai o áudio do MP4 e converte para um WAV não comprimido usando ffmpeg."""
    command = [
        "ffmpeg", "-y", "-i", str(video_path), 
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(wav_path)
    ]
    # Executa silenciosamente
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extrair_acustica(input_video, output_path):

    input_path = Path(input_video)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_video}")
        return

    print(f"[*] Extraindo áudio do vídeo {input_path.name}...")
    
    # Cria um arquivo temporário seguro para armazenar o WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name

    try:
        extract_audio_from_video(input_path, temp_wav_path)
        
        print("[*] Processando áudio no Parselmouth (Praat)...")
        # Carrega o áudio no objeto do Praat
        sound = parselmouth.Sound(temp_wav_path)
        
        # Extrai os objetos de Pitch (F0) e Intensidade
        pitch = sound.to_pitch()
        intensity = sound.to_intensity()
        
        # O Praat gera amostras em pequenos frames de tempo.
        # xs() retorna um array com os tempos exatos (em segundos) de cada frame analisado.
        pitch_times = pitch.xs()
        
        print(f"[*] Gerando arquivo {output_path}...")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp_ms', 'pitch_hz', 'intensity_db'])
            
            for t in pitch_times:
                # Busca o valor no tempo 't' exato
                p_val = pitch.get_value_at_time(t)
                i_val = intensity.get_value(t)
                
                # Tratamento de dados: 
                # Quando a pessoa faz pausas ou produz sons surdos (como 's', 'f'), não há frequência fundamental.
                # O Praat retorna 'NaN' (Not a Number). Trocamos isso por 0.0 para não quebrar o CSV.
                p_clean = round(p_val, 2) if not np.isnan(p_val) else 0.0
                i_clean = round(i_val, 2) if not np.isnan(i_val) else 0.0
                
                # Salva o tempo multiplicado por 1000 para bater perfeitamente com a lógica do MediaPipe
                writer.writerow([round(t * 1000, 2), p_clean, i_clean])
                
        print(f"[OK] Extração concluída! {len(pitch_times)} amostras acústicas salvas em {output_path}")

    finally:
        # Boas práticas: deleta o arquivo WAV temporário do seu HD no final do processo
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
