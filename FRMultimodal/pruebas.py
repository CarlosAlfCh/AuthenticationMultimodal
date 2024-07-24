import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time
import tkinter as tk
from tkinter import messagebox
import random

def check_password(stored_password):
    # Sonidos
    sound = pyglet.media.load("C:/prueba/AutheMultimodal/FRMultimodal/src/sound.wav", streaming=False)
    cap = cv2.VideoCapture(0)
    # Ajustes del teclado y del tablero
    keyboard_height = 400  # Aumentar el alto del teclado
    keyboard_width = 600
    board_height = 80
    board_width = 600

    keyboard = np.zeros((keyboard_height, keyboard_width, 3), np.uint8)
    board = np.zeros((board_height, board_width), np.uint8)
    board[:] = 255

    detector = dlib.get_frontal_face_detector()
    predictor_path = "C:/prueba/AutheMultimodal/FRMultimodal/src/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)

    # Generar números aleatorios
    random_numbers = list(range(10))
    random.shuffle(random_numbers)

    # Opciones del teclado con números aleatorios
    keys_set_1 = {
        0: str(random_numbers[0]), 1: str(random_numbers[1]), 2: str(random_numbers[2]),
        3: str(random_numbers[3]), 4: str(random_numbers[4]), 5: str(random_numbers[5]),
        6: str(random_numbers[6]), 7: str(random_numbers[7]), 8: str(random_numbers[8]),
        9: "Salir", 10: str(random_numbers[9]), 11: "Borrar Todo"
    }

    def letter(letter_index, text, letter_light):
        if letter_index < 9:
            x = (letter_index % 3) * 200
            y = (letter_index // 3) * 100  # Ajusta la altura de cada tecla
        else:
            x = ((letter_index - 9) % 3) * 200
            y = 300  # Nueva fila para el 9 y los botones A, B

        width = 200
        height = 100
        th = 3  # thickness

        # Fondo celeste claro para teclas inactivas, azul brillante para la activa
        if letter_light:
            cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 207, 119), -1)
        else:
            cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (248, 194, 111), -1)

        # Borde azul oscuro para todas las teclas

        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (81, 35, 10), th)

        font_letter = cv2.FONT_HERSHEY_PLAIN
        if letter_index < 9 or letter_index == 10:
            font_scale = 5
            font_th = 4
        else:
            font_scale = 1.5
            font_th = 2
        text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
        width_text, height_text = text_size[0], text_size[1]
        text_x = int((width - width_text) / 2) + x
        text_y = int((height + height_text) / 2) + y

        # Texto blanco para la tecla activa, negro para las inactivas
        text_color = (69, 71, 70) if letter_light else (0, 0, 0)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, text_color, font_th)

    # Definir la fuente que se usará
    font = cv2.FONT_HERSHEY_PLAIN

    # Funciones para calcular el ratio de parpadeo y el ratio de la mirada
    def midpoint(p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


    def get_blinking_ratio(eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

        horizontal_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
        vertical_length = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])

        ratio = horizontal_length / vertical_length
        return ratio


    def get_gaze_ratio(eye_points, facial_landmarks):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)

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


    # Contadores
    frames = 0
    letter_index = 0
    blinking_frames = 0
    text = ""
    speed = 30  # Ajusta esta variable para controlar la velocidad de cambio de números

    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        keyboard[:] = (110, 58, 26)  # Fondo azul oscuro suave
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
                cv2.putText(frame, "BLINKING", (30, 50), font, 1, (255, 0, 0), 2)
                blinking_frames += 1

                if blinking_frames == 5:
                    if active_letter == "Salir":
                        sound.play()
                    elif active_letter == "Borrar Todo":
                        text = ""  # Borrar todos los dígitos
                        sound.play()
                    elif len(text) < 4:  # Limitar a 4 números
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

        if letter_index == 12:
            letter_index = 0

        for i in range(12):
            light = i == letter_index
            letter(i, keys_set_1[i], light)

        # Espacio de entrada de texto
        board[:] = 255  # Limpiar el espacio de entrada de texto
        display_text = '*' * len(text)  # Mostrar asteriscos en lugar de los números reales
        text_size = cv2.getTextSize(display_text, font, 4, 3)[0]
        text_x = (board_width - text_size[0]) // 2
        text_y = (board_height + text_size[1]) // 2
        cv2.putText(board, display_text, (text_x, text_y), font, 4, (0, 0, 0), 3)

        # Agregar el texto "Crea tu Patrón Ocular" encima del recuadro blanco
        combined_frame = np.zeros((keyboard_height + board_height + 50, keyboard_width, 3), np.uint8)
        combined_frame[:] = 255  # Fondo blanco para el área adicional
        cv2.putText(combined_frame, "Ingrese su clave de 4 digitos", (50, 40), font, 2, (10, 35, 81), 3)  # Azul oscuro

        # Combina el frame, teclado y el tablero
        combined_frame[50:50 + board_height, :board_width] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        combined_frame[50 + board_height:, ] = keyboard

        cv2.imshow('frame', frame)
        cv2.imshow("Virtual keyboard", combined_frame)

        if text == stored_password:  # Comparar con la contraseña almacenada
            print("Contrasena correcta")
            return True
        elif len(text) == 4 and text != stored_password:  # Si se ingresaron 4 dígitos y no coincide
            print("Contrasena incorrecta")
            text = ""  # Reiniciar el texto ingresado

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return False