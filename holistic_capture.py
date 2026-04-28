"""
MediaPipe Holistic Capture
==========================
Captura face landmarks, pose, mãos esquerda e direita via câmera USB.
- Exibe o frame com os desenhos dos landmarks em tempo real
- Salva as coordenadas em um arquivo CSV a cada frame processado

Uso:
    python holistic_capture.py [--camera 0] [--output saida.csv] [--width 1280] [--height 720]

Dependências:
    pip install mediapipe opencv-python
"""

import cv2
import csv
import time
import argparse
import mediapipe as mp
from pathlib import Path

# ─── MediaPipe setup ───────────────────────────────────────────────────────────
mp_holistic   = mp.solutions.holistic
mp_drawing    = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── Helpers ───────────────────────────────────────────────────────────────────

def build_csv_header() -> list[str]:
    """Constrói o cabeçalho do CSV com todos os landmarks esperados."""
    header = ["frame", "timestamp_s"]

    # Face mesh: 468 landmarks
    for i in range(468):
        for axis in ("x", "y", "z"):
            header.append(f"face_{i}_{axis}")

    # Pose: 33 landmarks + visibility
    for i in range(33):
        for axis in ("x", "y", "z", "visibility"):
            header.append(f"pose_{i}_{axis}")

    # Mão esquerda: 21 landmarks
    for i in range(21):
        for axis in ("x", "y", "z"):
            header.append(f"left_hand_{i}_{axis}")

    # Mão direita: 21 landmarks
    for i in range(21):
        for axis in ("x", "y", "z"):
            header.append(f"right_hand_{i}_{axis}")

    return header


def extract_face(results) -> list:
    """Extrai coordenadas de face_landmarks (468 pontos × 3 eixos)."""
    row = []
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]
    else:
        row += [None] * (468 * 3)
    return row


def extract_pose(results) -> list:
    """Extrai coordenadas de pose_landmarks (33 pontos × 4 valores)."""
    row = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]
    else:
        row += [None] * (33 * 4)
    return row


def extract_hand(landmarks) -> list:
    """Extrai coordenadas de uma mão (21 pontos × 3 eixos)."""
    if landmarks:
        return [v for lm in landmarks.landmark for v in (lm.x, lm.y, lm.z)]
    return [None] * (21 * 3)


def draw_landmarks(frame, results):
    """Desenha todos os landmarks no frame."""
    # Face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    # Mão esquerda
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_utils_spec(color=(121, 22, 76)),
            mp_drawing_utils_spec(color=(121, 44, 250), thickness=2),
        )

    # Mão direita
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_utils_spec(color=(245, 117, 66)),
            mp_drawing_utils_spec(color=(245, 66, 230), thickness=2),
        )


def mp_drawing_utils_spec(color=(255, 255, 255), thickness=1, circle_radius=2):
    return mp_drawing.DrawingSpec(color=color, thickness=thickness, circle_radius=circle_radius)


def overlay_info(frame, frame_idx: int, fps: float, csv_path: str):
    """Sobrepõe informações de status no frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Frame: {frame_idx:05d}  |  FPS: {fps:5.1f}  |  CSV: {csv_path}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, "Pressione Q para sair | ESPACO para pausar",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Holistic Capture")
    parser.add_argument("--camera",  type=int,   default=0,            help="Índice da câmera USB (padrão: 0)")
    parser.add_argument("--output",  type=str,   default="holistic_landmarks.csv", help="Nome do arquivo CSV de saída")
    parser.add_argument("--width",   type=int,   default=1280,         help="Largura do frame (padrão: 1280)")
    parser.add_argument("--height",  type=int,   default=720,          help="Altura do frame (padrão: 720)")
    parser.add_argument("--min_det", type=float, default=0.5,          help="Confiança mínima de detecção (0-1)")
    parser.add_argument("--min_trk", type=float, default=0.5,          help="Confiança mínima de tracking (0-1)")
    args = parser.parse_args()

    csv_path = Path(args.output)
    header   = build_csv_header()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera índice {args.camera}.")
        return

    print(f"[INFO] Câmera {args.camera} aberta — {args.width}×{args.height}")
    print(f"[INFO] Salvando landmarks em: {csv_path.resolve()}")
    print("[INFO] Q = sair | ESPAÇO = pausar/retomar")

    frame_idx = 0
    paused    = False
    fps_vals  = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        with mp_holistic.Holistic(
            min_detection_confidence=args.min_det,
            min_tracking_confidence=args.min_trk,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
        ) as holistic:

            t_prev = time.perf_counter()

            while cap.isOpened():
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[INFO] Encerrando por solicitação do usuário.")
                    break

                if key == ord(" "):
                    paused = not paused
                    print("[INFO] " + ("Pausado." if paused else "Retomado."))

                if paused:
                    continue

                ret, frame = cap.read()
                if not ret:
                    print("[AVISO] Frame não capturado — tentando novamente...")
                    continue

                # Flip horizontal (espelho) para experiência mais natural
                frame = cv2.flip(frame, 1)

                # MediaPipe precisa de RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # Calcular FPS
                t_now = time.perf_counter()
                elapsed = t_now - t_prev
                t_prev  = t_now
                fps_inst = 1.0 / elapsed if elapsed > 0 else 0.0
                fps_vals.append(fps_inst)
                if len(fps_vals) > 30:
                    fps_vals.pop(0)
                fps_avg = sum(fps_vals) / len(fps_vals)

                # Timestamp relativo ao início (primeira captura)
                if frame_idx == 0:
                    t_start = t_now
                timestamp = t_now - t_start

                # ── Gravar CSV ─────────────────────────────────────────────
                row = [frame_idx, round(timestamp, 6)]
                row += extract_face(results)
                row += extract_pose(results)
                row += extract_hand(results.left_hand_landmarks)
                row += extract_hand(results.right_hand_landmarks)
                writer.writerow(row)

                # ── Desenhar e exibir ──────────────────────────────────────
                draw_landmarks(frame, results)
                overlay_info(frame, frame_idx, fps_avg, str(csv_path))

                cv2.imshow("MediaPipe Holistic — Pressione Q para sair", frame)
                frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] {frame_idx} frames processados. CSV salvo em: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
