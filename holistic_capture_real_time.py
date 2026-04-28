"""
MediaPipe Holistic Capture
==========================
Captura face landmarks, pose, mãos esquerda e direita via câmera USB.
- Exibe o frame com os desenhos dos landmarks em tempo real
- Salva as coordenadas em CSV via thread separada (sem bloquear o loop principal)

Arquitetura de threads:
  Thread principal  → captura + inferência MediaPipe + exibição OpenCV
  Thread escritora  → consome a fila e grava no CSV (I/O desacoplado)

Uso:
    python holistic_capture.py [--camera 0] [--output saida.csv] [--width 1280] [--height 720]

Dependências:
    pip install mediapipe==0.10.14 opencv-python
"""

import cv2
import csv
import time
import queue
import argparse
import threading
import mediapipe as mp
from pathlib import Path

# ─── MediaPipe setup ───────────────────────────────────────────────────────────
mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Sentinel para sinalizar fim da fila à thread escritora
_STOP = object()


# ─── Thread escritora de CSV ────────────────────────────────────────────────────

def csv_writer_thread(csv_path: Path, header: list, row_queue: queue.Queue):
    """
    Roda em thread separada.
    Consome rows da fila e as grava no CSV em lotes para minimizar syscalls.
    """
    BATCH = 30  # grava em lote a cada N rows (≈ 1 segundo a 30 fps)

    with open(csv_path, "w", newline="", encoding="utf-8", buffering=1 << 20) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        batch = []

        while True:
            item = row_queue.get()

            if item is _STOP:
                # Drena o restante da fila antes de encerrar
                while not row_queue.empty():
                    leftover = row_queue.get_nowait()
                    if leftover is not _STOP:
                        batch.append(leftover)
                if batch:
                    writer.writerows(batch)
                f.flush()
                break

            batch.append(item)
            if len(batch) >= BATCH:
                writer.writerows(batch)
                batch.clear()


# ─── Helpers de extração ────────────────────────────────────────────────────────

def build_csv_header() -> list:
    header = ["frame", "timestamp_s"]
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


def extract_face(results) -> list:
    if results.face_landmarks:
        return [v for lm in results.face_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
    return [None] * (468 * 3)


def extract_pose(results) -> list:
    if results.pose_landmarks:
        return [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)]
    return [None] * (33 * 4)


def extract_hand(landmarks) -> list:
    if landmarks:
        return [v for lm in landmarks.landmark for v in (lm.x, lm.y, lm.z)]
    return [None] * (21 * 3)


def build_row(frame_idx: int, timestamp: float, results) -> list:
    """Monta a row completa em memória — sem I/O, chamado na thread principal."""
    row = [frame_idx, round(timestamp, 6)]
    row += extract_face(results)
    row += extract_pose(results)
    row += extract_hand(results.left_hand_landmarks)
    row += extract_hand(results.right_hand_landmarks)
    return row


# ─── Helpers de visualização ────────────────────────────────────────────────────

def _spec(color, thickness=1, radius=2):
    return mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=radius)


def draw_landmarks(frame, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            _spec((121, 22, 76)), _spec((121, 44, 250), thickness=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            _spec((245, 117, 66)), _spec((245, 66, 230), thickness=2),
        )


def overlay_info(frame, frame_idx: int, fps: float, q_size: int, csv_name: str):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        f"Frame: {frame_idx:05d}  FPS: {fps:5.1f}  Fila CSV: {q_size:04d}  {csv_name}",
        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Q = sair  |  ESPACO = pausar",
        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Holistic Capture")
    parser.add_argument("--camera",    type=int,   default=0)
    parser.add_argument("--output",    type=str,   default="holistic_landmarks_realtime_results.csv")
    parser.add_argument("--width",     type=int,   default=640)
    parser.add_argument("--height",    type=int,   default=480)
    parser.add_argument("--min_det",   type=float, default=0.5)
    parser.add_argument("--min_trk",   type=float, default=0.5)
    parser.add_argument("--queue_max", type=int,   default=500,
                        help="Tamanho máximo da fila (padrão 500). "
                             "Rows além desse limite são descartadas silenciosamente.")
    args = parser.parse_args()

    csv_path = Path(args.output)
    header   = build_csv_header()

    # Fila bounded: impede que RAM cresça indefinidamente se disco for lento
    row_queue: queue.Queue = queue.Queue(maxsize=args.queue_max)

    # Thread escritora sobe antes da câmera para não perder nenhum frame
    writer_thread = threading.Thread(
        target=csv_writer_thread,
        args=(csv_path, header, row_queue),
        daemon=False,   # não-daemon garante flush mesmo em crash do main
        name="CSVWriter",
    )
    writer_thread.start()

    win_name = "MediaPipe Holistic"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) # Permite redimensionar
    cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO) # Mantém proporção

    cap = cv2.VideoCapture(args.camera)
    
    # Tenta definir a resolução, mas lê o que a câmera REALMENTE entregou
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.resizeWindow(win_name, 1280, 720)


    if not cap.isOpened():
        print(f"[ERRO] Câmera {args.camera} não encontrada.")
        row_queue.put(_STOP)
        writer_thread.join()
        return

    print(f"[INFO] Câmera iniciada: {actual_w}x{actual_h}")
    print(f"[INFO] CSV: {csv_path.resolve()}")
    print("[INFO] Q = sair | ESPAÇO = pausar/retomar")

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
    # Opcional: define um tamanho inicial para a janela
    cv2.resizeWindow(win_name, 1280, 720)

    frame_idx = 0
    paused    = False
    dropped   = 0
    fps_vals  = []
    t_start   = None

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=args.min_det,
            min_tracking_confidence=args.min_trk,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:

            t_prev = time.perf_counter()

            while cap.isOpened():
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[INFO] Encerrando.")
                    break
                if key == ord(" "):
                    paused = not paused
                    print("[INFO] " + ("Pausado." if paused else "Retomado."))

                if paused:
                    continue

                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                # ── Inferência MediaPipe ───────────────────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # ── FPS ────────────────────────────────────────────────────────
                t_now   = time.perf_counter()
                elapsed = t_now - t_prev
                t_prev  = t_now
                fps_vals.append(1.0 / elapsed if elapsed > 0 else 0.0)
                if len(fps_vals) > 30:
                    fps_vals.pop(0)
                fps_avg = sum(fps_vals) / len(fps_vals)

                if t_start is None:
                    t_start = t_now
                timestamp = t_now - t_start

                # ── Enfileirar row (put_nowait = nunca trava o loop principal) ─
                row = build_row(frame_idx, timestamp, results)
                try:
                    row_queue.put_nowait(row)
                except queue.Full:
                    dropped += 1  # disco lento: descarta frame, não trava a câmera

                # ── Visualização ───────────────────────────────────────────────
                draw_landmarks(frame, results)
                overlay_info(frame, frame_idx, fps_avg, row_queue.qsize(), csv_path.name)
                cv2.imshow(win_name, frame)
                

                frame_idx += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário.")
    
    finally:
        # ── Encerramento limpo ──────────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()

        if writer_thread.is_alive():
            pending = row_queue.qsize()
            print(f"[INFO] Aguardando escrita de {pending} rows pendentes no CSV...")
            row_queue.put(_STOP)
            writer_thread.join(timeout=3)

        print(f"[OK] {frame_idx} frames | {dropped} descartados | CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
