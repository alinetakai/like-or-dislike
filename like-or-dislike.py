import cv2
import mediapipe as mp

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# Desenho dos landmarks
mp_drawing = mp.solutions.drawing_utils

# Inicializando a câmera
cap = cv2.VideoCapture(0)

# IDs das pontas dos dedos
tip_ids = [8, 12, 16, 20]

# ID da ponta do polegar
thumb_tip = 4

# ID da base/intermediário do polegar
thumb_base = 2

while True:

    # Captura da imagem da câmera
    ret, img = cap.read()

    if not ret:
        print("Não foi possível capturar a imagem da câmera.")
        break

    # Espelhando a imagem horizontalmente
    img = cv2.flip(img, 1)

    # Dimensões da imagem
    h, w, c = img.shape

    # Conversão de BGR para RGB para o MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processamento da imagem
    results = hands.process(img_rgb)

    # Verificando se alguma mão foi detectada
    if results.multi_hand_landmarks:

        for hand_landmark in results.multi_hand_landmarks:

            # Lista para armazenar os landmarks
            lm_list = []

            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Verificando as pontas dos outros dedos
            for tip in tip_ids:

                x = int(lm_list[tip].x * w)
                y = int(lm_list[tip].y * h)

                # Círculo padrão nas pontas dos dedos
                cv2.circle(
                    img,
                    (x, y),
                    15,
                    (203, 192, 255),
                    cv2.FILLED
                )

                # Verificando posição do dedo
                if lm_list[tip].x < lm_list[tip - 3].x:

                    cv2.circle(
                        img,
                        (x, y),
                        15,
                        (15, 216, 216),
                        cv2.FILLED
                    )

            # -----------------------------------
            # IDENTIFICAÇÃO DO GESTO DO POLEGAR
            # -----------------------------------

            thumb_tip_y = lm_list[thumb_tip].y
            thumb_base_y = lm_list[thumb_base].y

            if thumb_tip_y < thumb_base_y:

                print("Curti")

                cv2.putText(
                    img,
                    "Curti",
                    (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 85),
                    3
                )

            elif thumb_tip_y > thumb_base_y:

                print("Nao curti")

                cv2.putText(
                    img,
                    "Nao curti",
                    (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 85),
                    3
                )

            # Desenhando os landmarks e conexões da mão
            mp_drawing.draw_landmarks(
                img,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 0, 255),
                    thickness=2,
                    circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=4,
                    circle_radius=2
                )
            )

    # -----------------------------------
    # EXIBINDO A IMAGEM
    # -----------------------------------

    cv2.imshow("Hands tracking", img)

    # Pressione Q para encerrar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------------
# ENCERRANDO O PROGRAMA
# -----------------------------------

cap.release()
hands.close()
cv2.destroyAllWindows()