import cv2
import csv
import queue
import argparse
import threading
import mediapipe as mp
from pathlib import Path

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
    header = ["frame", "timestamp_ms"] # Mudado para ms para precisão de vídeo
    for i in range(468):
        for ax in ("x", "y", "z"): 
            header.append(f"face_{i}_{ax}")
    for i in range(33):
        for ax in ("x", "y", "z", "visibility"): 
            header.append(f"pose_{i}_{ax}")
    for i in range(21):
        for ax in ("x", "y", "z"): 
            header.append(f"left_hand_{i}_{ax}")
    for i in range(21):
        for ax in ("x", "y", "z"): 
            header.append(f"right_hand_{i}_{ax}")
    return header

def extract_landmarks(results):
    """Extrai todos os dados de uma vez."""
    face = [v for lm in results.face_landmarks.landmark for v in (lm.x, lm.y, lm.z)] if results.face_landmarks else [None]*1404
    pose = [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)] if results.pose_landmarks else [None]*132
    lh = [v for lm in results.left_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)] if results.left_hand_landmarks else [None]*63
    rh = [v for lm in results.right_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)] if results.right_hand_landmarks else [None]*63
    return face + pose + lh + rh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Caminho do vídeo")
    parser.add_argument("--output", type=str, default="holistic_landmarks_video_file_results.csv")
    parser.add_argument("--complexity", type=int, default=1, help="Complexidade do modelo: 0 (lite), 1 (normal) ou 2 (slow)")
    parser.add_argument("--min_det", type=float, default=0.5, help="Confiança mínima para detecção")
    parser.add_argument("--min_trk", type=float, default=0.5, help="Confiança mínima para rastreamento")
    parser.add_argument("--width", type=int, default=640, help="Largura do vídeo para processamento")
    parser.add_argument("--height", type=int, default=480, help="Altura do vídeo para processamento")
    parser.add_argument("--draw", action="store_true", help="Desenhar landmarks no vídeo")

    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"[ERRO] Vídeo não encontrado: {args.input}")
        return

    cap = cv2.VideoCapture(args.input)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup CSV
    row_queue = queue.Queue()
    writer_thread = threading.Thread(target=csv_writer_thread, args=(Path(args.output), build_csv_header(), row_queue))
    writer_thread.start()

    win_name = "Processando Vídeo..."
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    with mp_holistic.Holistic(
        model_complexity=args.complexity,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_trk,
        refine_face_landmarks=False
    ) as holistic:
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break # Fim do vídeo

            # Processamento
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Timestamp baseado no frame rate do vídeo (mais preciso que relógio do sistema)
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Salvar dados (sem put_nowait para garantir que não perca nenhum frame)
            row = [frame_idx, round(timestamp_ms, 2)] + extract_landmarks(results)
            row_queue.put(row)

            if args.draw:
                # Desenha a cada 2 frames para economizar CPU, mas processa TODOS
                if frame_idx % 2 == 0: 
                    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
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
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Processando: {progress:.1f}% concluído...", end="\r")

            frame_idx += 1
    

    cap.release()
    cv2.destroyAllWindows()
    
    row_queue.put(_STOP)
    writer_thread.join()
    print(f"\n[OK] Processamento concluído: {frame_idx} frames salvos em {args.output}")

if __name__ == "__main__":
    main()