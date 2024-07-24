import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

def check_password(stored_password):
    # Sonidos
    sound = pyglet.media.load("C:/prueba/AutheMultimodal/FRMultimodal/src/sound.wav", streaming=False)

    cap = cv2.VideoCapture(0)

    # Ajustes del teclado y del tablero
    keyboard_height = 600
    keyboard_width = 600
    board_height = 100
    board_width = 600

    keyboard = np.zeros((keyboard_height, keyboard_width, 3), np.uint8)
    board = np.zeros((board_height, board_width), np.uint8)
    board[:] = 255

    detector = dlib.get_frontal_face_detector()
    predictor_path = "C:/prueba/AutheMultimodal/FRMultimodal/src/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Opciones del teclado
    keys_set_1 = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5",
                  5: "6", 6: "7", 7: "8", 8: "9"}

    def letter(letter_index, text, letter_light):
        x = (letter_index % 3) * 200
        y = (letter_index // 3) * 200
        width = 200
        height = 200
        th = 3  # thickness

        # Fondo blanco y borde negro para todas las teclas
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 0), th)

        font_letter = cv2.FONT_HERSHEY_PLAIN
        font_scale = 10
        font_th = 4
        text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
        width_text, height_text = text_size[0], text_size[1]
        text_x = int((width - width_text) / 2) + x
        text_y = int((height + height_text) / 2) + y

        # Texto gris claro para la letra activa, negro para las demás
        text_color = (200, 200, 200) if letter_light else (0, 0, 0)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, text_color, font_th)

    def midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    font = cv2.FONT_HERSHEY_PLAIN

    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        ratio = hor_line_length / ver_line_length
        return ratio

    def get_gaze_ratio(eye_points, facial_landmarks):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white
        return gaze_ratio

    frames = 0
    letter_index = 0
    blinking_frames = 0
    text = ""
    speed = 35

    while True:
        _, frame = cap.read()
        keyboard[:] = (0, 0, 0)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame = cv2.flip(frame, 1)
        new_frame = np.zeros((500, 500, 3), np.uint8)
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        active_letter = keys_set_1[letter_index]

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > 5.7:
                cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
                blinking_frames += 1

                if blinking_frames == 5:
                    if len(text) < 4:  # Limitar a 4 números
                        text += active_letter
                        sound.play()
                        time.sleep(1)
                    blinking_frames = 0
                    frames = 0
            else:
                blinking_frames = 0

            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if frames == speed:
            letter_index += 1
            frames = 0

        if letter_index == 9:
            letter_index = 0

        for i in range(9):
            light = i == letter_index
            letter(i, keys_set_1[i], light)

        # Espacio de entrada de texto
        board[:] = 255  # Limpiar el espacio de entrada de texto
        cv2.putText(board, text, (10, 80), font, 4, (0, 0, 0), 3)

        # Combina el frame, teclado y el tablero
        combined_frame = np.zeros((keyboard_height + board_height, keyboard_width, 3), np.uint8)
        combined_frame[:board_height, :board_width] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        combined_frame[board_height:, ] = keyboard

        cv2.imshow('frame', frame)
        cv2.imshow("Virtual keyboard", combined_frame)

        if text == stored_password:  # Comparar con la contraseña almacenada
            print("Contraseña correcta")
            return True
        elif len(text) == 4 and text != stored_password:  # Si se ingresaron 4 dígitos y no coincide
            print("Contraseña incorrecta")
            text = ""  # Reiniciar el texto ingresado

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return False
