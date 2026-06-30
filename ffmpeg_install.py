import urllib.request
import zipfile
import os

def baixar_ffmpeg_standalone():
    if os.path.exists("ffmpeg.exe"):
        print("[OK] FFmpeg estático já está na pasta.")
        return

    print("[*] Baixando FFmpeg estático (imune a erros de DLL)...")
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    arquivo_zip = "ffmpeg_temp.zip"
    
    try:
        urllib.request.urlretrieve(url, arquivo_zip)
        print("[*] Extraindo o executável...")
        
        with zipfile.ZipFile(arquivo_zip, 'r') as z:
            for info in z.infolist():
                if info.filename.endswith("ffmpeg.exe"):
                    info.filename = "ffmpeg.exe"
                    z.extract(info, ".")
                    break
                    
        os.remove(arquivo_zip)
        print("[OK] FFmpeg configurado com sucesso!")
        
    except Exception as e:
        print(f"[ERRO] Falha ao baixar: {e}")

if __name__ == "__main__":
    baixar_ffmpeg_standalone()