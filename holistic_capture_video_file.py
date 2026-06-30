import cv2
import csv
import queue
import argparse
import threading
import mediapipe as mp
from pathlib import Path

# ─── Downsampling: 27 Landmarks do Artigo (FACS AUs) ───────────────────────────
SELECTED_FACE_INDICES = [
    61, 292,   # Cantos da boca
    0, 17,     # Centro do lábio superior/inferior
    50, 280,   # Bochechas (corrigido do typo do artigo)
    48, 4, 289,# Ponta e laterais do nariz
    206, 426,  # Mandíbula superior
    133, 130, 159, 145, 362, 359, 386, 374, # Cantos e pálpebras dos olhos
    122, 351,  # Ponte nasal
    46, 105, 107, 276, 334, 336 # Sobrancelhas
]

# ─── MediaPipe setup ───────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

_STOP = object()

def csv_writer_thread(csv_path: Path, header: list, row_queue: queue.Queue):
    """Grava os dados no CSV conforme chegam na fila."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        while True:
            item = row_queue.get()
            if item is _STOP: break
            writer.writerow(item)

def build_csv_header() -> list:
    header = ["frame", "timestamp_ms"]
    
    # Adiciona apenas os 27 pontos faciais selecionados
    for idx in SELECTED_FACE_INDICES:
        for ax in ("x", "y", "z"): 
            header.append(f"face_{idx}_{ax}")
            
    # Mantém o resto do corpo (pose e mãos) inalterado
    for i in range(33):
        for ax in ("x", "y", "z", "visibility"): header.append(f"pose_{i}_{ax}")
    for i in range(21):
        for ax in ("x", "y", "z"): header.append(f"left_hand_{i}_{ax}")
    for i in range(21):
        for ax in ("x", "y", "z"): header.append(f"right_hand_{i}_{ax}")
    return header

def extract_landmarks(results):
    """Extrai todos os dados de uma vez."""
    face = [v for lm in results.face_landmarks.landmark for v in (lm.x, lm.y, lm.z)] if results.face_landmarks else [None]*1404
    pose = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)] if results.pose_landmarks else [None]*132
    lh = [v for lm in results.left_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)] if results.left_hand_landmarks else [None]*63
    rh = [v for lm in results.right_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)] if results.right_hand_landmarks else [None]*63
    return face + pose + lh + rh

def extract_holistic_landmarks(input_path, output_path, complexity=1, min_det=0.5, min_trk=0.5, width=None, height=None, draw=False):

    video_path = Path(input_path)
    if not video_path.exists():
        print(f"[ERRO] Vídeo não encontrado: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup CSV
    row_queue = queue.Queue()
    writer_thread = threading.Thread(target=csv_writer_thread, args=(Path(output_path), build_csv_header(), row_queue))
    writer_thread.start()

    win_name = "Processando Vídeo..."
    if draw:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        target_width = 800
        if orig_w > target_width:
            target_height = int(orig_h * (target_width / orig_w))
        else:
            target_width = int(orig_w)
            target_height = int(orig_h)
    
        cv2.resizeWindow(win_name, target_width, target_height)
    

    try:
        with mp_holistic.Holistic(
            model_complexity=complexity,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk,
            refine_face_landmarks=False
        ) as holistic:
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break # Fim do vídeo

                if width is not None and height is not None:
                    frame = cv2.resize(frame, (width, height))
                
                # Processamento
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Timestamp baseado no frame rate do vídeo (mais preciso que relógio do sistema)
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Salvar dados (sem put_nowait para garantir que não perca nenhum frame)
                row = [frame_idx, round(timestamp_ms, 2)] + extract_landmarks(results)
                row_queue.put(row)

                if draw:
                    # Desenha a cada 2 frames para economizar CPU, mas processa TODOS
                    if frame_idx % 2 == 0: 
                        if results.face_landmarks:
                            h, w, c = frame.shape
                            for idx in SELECTED_FACE_INDICES:
                                lm = results.face_landmarks.landmark[idx]
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                # Desenha pequenos círculos verdes nos 27 pontos
                                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        if results.left_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        if results.right_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        
                        progress = (frame_idx / total_frames) * 100
                        cv2.putText(frame, f"Progresso: {progress:.1f}%", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow(win_name, frame)
            
                    # O waitKey precisa estar dentro ou logo após o imshow para a janela funcionar
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        break
                else:
                    # Atualiza a barra de progresso a cada 5 frames
                    if frame_idx % 5 == 0 or frame_idx == total_frames - 1:
                        percent = frame_idx / total_frames
                        bar_length = 40
                        filled = int(bar_length * percent)
                        bar = '█' * filled + '-' * (bar_length - filled)
                        # O flush=True garante que o terminal atualize a linha imediatamente
                        print(f"\rProcessando Vídeo: |{bar}| {percent*100:.1f}% ({frame_idx}/{total_frames})", end="", flush=True)

                frame_idx += 1
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        row_queue.put(_STOP)
        writer_thread.join()
        print(f"\n[OK] Processamento concluído: {frame_idx} frames salvos em {output_path}")