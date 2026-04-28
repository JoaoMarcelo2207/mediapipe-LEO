''' Esse algoritmo implementa uma forma simplória para detecção de sorriso, ele identifica apenas se a largura da boca aumenta ao calcular a distância euclidiana entre os pontos do canto da boca. Como ao utilizar landmarks estamos considerando perspectivas, é importante normalizar a distância com a algum ponto do rosto, pois se olharmos apenas para a boca, movimentos para frente e para trás do rosto em relação à camera caracterizam um aumento na distância entre os cantos da boca, mas que não são decorrente de uma expressão facial.'''

import cv2
import mediapipe as mp
import math

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_smile_ratio(landmarks, w, h):
    # landmarks usados
    LEFT_MOUTH  = 61
    RIGHT_MOUTH = 291
    LEFT_FACE   = 234
    RIGHT_FACE  = 454

    def lp(i):
        return (landmarks[i].x * w, landmarks[i].y * h)

    # largura da boca
    mouth_width = dist(lp(LEFT_MOUTH), lp(RIGHT_MOUTH))

    # largura da face
    face_width = dist(lp(LEFT_FACE), lp(RIGHT_FACE))

    if face_width == 0:
        return 0

    # normalização: valor entre 0 e 1
    smile_ratio = mouth_width / face_width

    return smile_ratio


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=True
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # converter BGR → RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # processar com holistic
        results = holistic.process(img)

        # voltar RGB → BGR para exibir
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # dimensões da imagem
        h, w, _ = img.shape

        # desenhar landmarks
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        # detectar sorriso
        if results.face_landmarks:
            ratio = get_smile_ratio(results.face_landmarks.landmark, w, h)
            print(ratio)

            if ratio > 0.39:
                cv2.putText(img, "SORRISO :)",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0,255,0), 3)
            else:
                cv2.putText(img, "SEM SORRISO",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0,0,255), 3)

        # exibir
        cv2.imshow('Holistic', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
