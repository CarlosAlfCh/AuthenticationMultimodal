import cv2
import dlib
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Cargar el detector de caras de dlib y el predictor de puntos de referencia facial
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/prueba/AutheMultimodal/FRMultimodal/src/shape_predictor_68_face_landmarks.dat")


# Función para calcular la relación de aspecto del ojo (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Definir constantes para detectar parpadeos
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Contador de marcos y número total de parpadeos
COUNTER = 0
TOTAL = 0

# Listas para el gráfico
time_series = []

# Configurar el gráfico
plt.style.use('seaborn')
fig, ax = plt.subplots()
line, = ax.plot(time_series, color='red')
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, 60)
ax.set_xlabel('Tiempo transcurrido')
ax.set_ylabel('Parpadeos')
ax.set_title('Detector de parpadeos')


def update_graph(frame):
    global COUNTER, TOTAL

    ret, frame = cap.read()
    if not ret:
        return line,

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    blink_detected = False

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extraer coordenadas de los ojos
        leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        rightEye = shape[
                   face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

        # Calcular el EAR para ambos ojos
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Visualizar puntos de referencia facial
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Aumentar el radio a 3

        # Comprobar si el EAR está por debajo del umbral para detectar parpadeos
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            blink_detected = True
        else:
            COUNTER = 0

    if blink_detected:
        time_series.append(1)
        cv2.putText(frame, "Ojo Cerrado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        time_series.append(0)

    if len(time_series) > 50:
        time_series.pop(0)

    # Mostrar el frame con los puntos de referencia y el conteo de parpadeos
    cv2.imshow("Camara", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        plt.close()

    # Actualizar el gráfico
    line.set_ydata(time_series)
    line.set_xdata(range(len(time_series)))
    ax.set_xlim(max(0, len(time_series) - 50), max(50, len(time_series)))
    return line,


# Iniciar captura de video
cap = cv2.VideoCapture(0)

# Crear la animación del gráfico
ani = FuncAnimation(fig, update_graph, blit=True, interval=50)

# Mostrar el gráfico
plt.show()

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
