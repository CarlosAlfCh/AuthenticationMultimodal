from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import math
import os
import pruebas

#Login Biometrico
def logBiometric():
    global pantalla2, conteo, parpadeo, imgInfo, step, cap, lblVideo, RegUser
    if cap is not None:
        ret, frame = cap.read()

        frameSave = frame.copy()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # frame RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame show

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            res = faceMesh.process(frameRGB)
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                for rostros in res.multi_face_landmarks:
                    mpDraw.draw_landmarks(frame, rostros, faceMeshObject.FACEMESH_CONTOURS, configDraw, configDraw)
                    for id, puntos in enumerate(rostros.landmark):
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])
                        # 468 puntos
                        if len(lista)==468:
                            # ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2 - x1, y2 - y1)
                            # ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # verificar si mira a camara
                            x5, y5 = lista[139][1:]
                            x6, y6 = lista[368][1:]

                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            faces = detector.process(frameRGB)
                            if faces.detections is not None:
                                for face in faces.detections:
                                    #bbox : ID, BBOX, SCORE
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    if score > confThreshold:
                                        # conv a pixels
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        offsetan = (offsetx / 100) * anc
                                        xi = int(xi - int(offsetan / 2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        if step == 0:
                                            cv2. rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)

                                            # pasos
                                            # Centrar
                                            if x7 > x5 and x8 < x6:
                                                cv2.putText(frame, f'Parpadee para continuar', (70, 80),
                                                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (70, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                                                if conteo >= 1:
                                                    cv2.putText(frame, f'OK', (70, 500),
                                                                cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 3)

                                                    if longitud1 > 15 and longitud2 > 15:
                                                        cut = frameSave[yi:yf, xi:xf]
                                                        cv2.imwrite(f"{OutFolderPathFace}/{RegUser}.png", cut)
                                                        step = 1
                                                        # Finalizar la captura y volver a la pantalla inicial
                                                        Close_Window()


                                            else:
                                                cv2.putText(frame, f'Mire hacia el frente', (70, 200),
                                                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                                                conteo=0

                                        if step == 1:

                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            cv2.putText(frame, f'OK', (70, 500),
                                                        cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 3)

                            # ver puntos
                            #cv2.circle(frame, (x1, y1), 2, (255, 0, 0), cv2.FILLED)
                            #cv2.circle(frame, (x2, y2), 2, (255, 0, 0), cv2.FILLED)

        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        lblVideo.configure(image = img)
        lblVideo.image = img
        lblVideo.after(10, logBiometric)
    else:
        cap.release()

def Profile():
    global step, conteo, userName, OutFolderPathUser
    # Reset Variables
    conteo = 0
    step = 0

    pantalla4 = Toplevel(pantalla)
    pantalla4.title("BIOMETRIC SIGN")
    pantalla4.geometry("1280x720")

    back = Label(pantalla4, image=imagenIn, text="Back")
    back.place(x=0, y=0, relwidth=1, relheight=1)

    # Archivo
    UserFile = open(f"{OutFolderPathUser}/{userName}.txt", 'r')
    InfoUser = UserFile.read().split(',')
    Name = InfoUser[0]
    User = InfoUser[1]
    Pass = InfoUser[2]
    UserFile.close()

    # Verificar contraseña
    if pruebas.check_password(Pass):
        # Check
        if User in clases:
            # Interfaz
            texto1 = Label(pantalla4, text=f"BIENVENIDO {Name}", font=("Helvetica", 12, "bold") )
            texto1.place(x=30, y=40)
            # Label
            # Video
            lblImgUser = Label(pantalla4)
            lblImgUser.place(x=20, y=90)

            # Imagen
            PosUserImg = clases.index(User)
            UserImg = images[PosUserImg]

            ImgUser = Image.fromarray(UserImg)
            #
            ImgUser = cv2.imread(f"{OutFolderPathFace}/{User}.png")
            ImgUser = cv2.cvtColor(ImgUser, cv2.COLOR_RGB2BGR)
            ImgUser = Image.fromarray(ImgUser)
            #

            # Redimensionar la imagen a un tamaño específico (ancho, alto)
            ImgUser = ImgUser.resize((250, 275), Image.Resampling.LANCZOS)  # Cambia (150, 150) al tamaño deseado

            IMG = ImageTk.PhotoImage(image=ImgUser)

            lblImgUser.configure(image=IMG)
            lblImgUser.image = IMG
    else:
        print("Acceso denegado. Contraseña incorrecta.")


def Sign_Biometric():
    global logUser, logPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images, step, parpadeo, conteo, userName

    if cap is not None:
        ret, frame = cap.read()

        frameSave = frame.copy()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # frame RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame show

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            res = faceMesh.process(frameRGB)
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                for rostros in res.multi_face_landmarks:
                    mpDraw.draw_landmarks(frame, rostros, faceMeshObject.FACEMESH_CONTOURS, configDraw, configDraw)
                    for id, puntos in enumerate(rostros.landmark):
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])
                        # 468 puntos
                        if len(lista)==468:
                            # ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2 - x1, y2 - y1)
                            # ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # verificar si mira a camara
                            x5, y5 = lista[139][1:]
                            x6, y6 = lista[368][1:]

                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            faces = detector.process(frameRGB)
                            if faces.detections is not None:
                                for face in faces.detections:
                                    #bbox : ID, BBOX, SCORE
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    if score > confThreshold:
                                        # conv a pixels
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        offsetan = (offsetx / 100) * anc
                                        xi = int(xi - int(offsetan / 2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        if step == 0:
                                            cv2. rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)


                                            # pasos
                                            # Centrar
                                            if x7 > x5 and x8 < x6:
                                                cv2.putText(frame, f'Parpadee para continuar', (70, 80),
                                                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (70, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),2)

                                                if conteo >= 0:
                                                    cv2.putText(frame, f'OK', (70, 500),
                                                                cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 3)

                                                    if longitud1 > 15 and longitud2 > 15:
                                                        step = 1

                                            else:
                                                cv2.putText(frame, f'Mire hacia el frente', (70, 200),
                                                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                                                conteo=0

                                        if step == 1:

                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            cv2.putText(frame, f'OK', (70, 500),
                                                        cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)

                                            facess = fr.face_locations(frameRGB)
                                            facescod = fr.face_encodings(frameRGB, facess)

                                            for facescod, facesloc in zip(facescod, facess):
                                                match = fr.compare_faces(FaceCode, facescod)
                                                simi = fr.face_distance(FaceCode,facescod)
                                                min = np.argmin(simi)
                                                if match[min]:
                                                    userName = clases[min].upper()
                                                    Close_Window2()
                                                    Profile()

                            # ver puntos
                            #cv2.circle(frame, (x1, y1), 2, (255, 0, 0), cv2.FILLED)
                            #cv2.circle(frame, (x2, y2), 2, (255, 0, 0), cv2.FILLED)

        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        lblVideo.configure(image = img)
        lblVideo.image = img
        lblVideo.after(10, Sign_Biometric)
    else:
        cap.release()

def Close_Window():
    global pantalla2, pantalla, step, conteo
    conteo = 0
    step = 0
    pantalla2.destroy()
    pantalla.deiconify()

def Close_Window2():
    global pantalla3, step, conteo
    conteo = 0
    step = 0
    pantalla3.destroy()

def Code_Face(images):
    listacod = []

    # Iteramos
    for img in images:
        # Correccion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Codificamos la imagen
        cod = fr.face_encodings(img)[0]
        # Almacenamos
        listacod.append(cod)

    return listacod

def Log():
    global logUser, logPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images
    # DB Faces
    # Accedemos a la carpeta
    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    # Leemos los rostros del DB
    for lis in lista:
        # Leemos las imagenes de los rostros
        imgdb = cv2.imread(f'{OutFolderPathFace}/{lis}')
        # Almacenamos imagen
        images.append(imgdb)
        # Almacenamos nombre
        clases.append(os.path.splitext(lis)[0])

    # Face Code
    FaceCode = Code_Face(images)

    # 3° Ventana
    pantalla3 = Toplevel(pantalla)
    pantalla3.title("HOLA BIOMETRICO")
    pantalla3.geometry("1280x720")

    # Video
    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Sign_Biometric()

def Sign():
    global RegName, RegUser, RegPass, InputNameReg, InputUserReg, InputPassReg, cap, lblVideo, pantalla2
    RegName, RegUser, RegPass = InputNameReg.get(), InputUserReg.get(), InputPassReg.get()
    if len(RegName)==0 or len(RegUser)==0 or len(RegPass)==0:
        print("xd")
    else:
        userList = os.listdir(PathUserCheck)

        #Extraccion
        userName = []
        for lis in userList:
            user = lis
            user = user.split('.')
            userName.append(user[0])

        if RegUser in userName:
            print("USUARIO REGISTRADO ANTERIORMENTE")
        else:
            info.append(RegName)
            info.append(RegUser)
            info.append(RegPass)

            f = open(f"{OutFolderPathUser}/{RegUser}.txt", "w")
            f.write(RegName + ',')
            f.write(RegUser + ',')
            f.write(RegPass)
            f.close()
            # Limpiar espacios
            InputNameReg.delete(0,END)
            InputUserReg.delete(0,END)
            InputPassReg.delete(0,END)

            pantalla2 = Toplevel(pantalla)
            pantalla2.title("LOGIN BIOMETRICO")
            pantalla2.geometry("1250x720")

            #label de video
            lblVideo = Label(pantalla2)
            lblVideo.place(x=0, y=0)

            #captura de video
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(3, 1280)
            cap.set(4, 720)
            logBiometric()

#Path

OutFolderPathUser = 'C:/prueba/AutheMultimodal/FRMultimodal/DataBase/Users'
PathUserCheck = 'C:/prueba/AutheMultimodal/FRMultimodal/DataBase/Users/'
OutFolderPathFace = 'C:/prueba/AutheMultimodal/FRMultimodal/DataBase/Faces'

#imagenes
imgstep0= cv2.imread("C:/prueba/AutheMultimodal/FRMultimodal/src/step0.png")


# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0

offsety = 40
offsetx = 20

confThreshold = 0.5

mpDraw = mp.solutions.drawing_utils
configDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

faceMeshObject = mp.solutions.face_mesh
faceMesh = faceMeshObject.FaceMesh(max_num_faces=1)

#Object face detector
faceObject = mp.solutions.face_detection
detector = faceObject.FaceDetection(min_detection_confidence=0.5, model_selection=1)




# Info List
info = []

# Intefaces Principal
pantalla = Tk()
pantalla.title("Sistema de Reconocimiento Facial")
pantalla.geometry("1280x720")

# BackGraund
imagenF = PhotoImage(file="C:/prueba/AutheMultimodal/FRMultimodal/src/main.png")
imagenIn = PhotoImage(file="C:/prueba/AutheMultimodal/FRMultimodal/src/back11.png")

background = Label(image = imagenF, text = "Inicio")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)

# Inputs
# Etiqueta y Input para Nombre
labelName = Label(pantalla, text="Nombre:")
labelName.place(x=110, y=320)
InputNameReg = Entry(pantalla, width=60)  # Ancho triplicado
InputNameReg.place(x=110, y=350)

# Etiqueta y Input para Usuario
labelUser = Label(pantalla, text="Usuario:")
labelUser.place(x=110, y=400)
InputUserReg = Entry(pantalla, width=60)  # Ancho triplicado
InputUserReg.place(x=110, y=430)

# Etiqueta y Input para Contraseña
labelPass = Label(pantalla, text="Contraseña:")
labelPass.place(x=110, y=480)
InputPassReg = Entry(pantalla, width=60)  # Ancho triplicado
InputPassReg.place(x=110, y=510)

# Botones
# Botón de Registro
BtReg = Button(pantalla, text="Registrar", bg="blue", fg="white", height=2, width=20, command=Sign)
BtReg.place(x=300, y=580)

# Botón de Inicio de Sesión
BtSign = Button(pantalla, text="Iniciar Sesión", bg="green", fg="white", height=2, width=20, command=Log)
BtSign.place(x=850, y=580)

pantalla.mainloop()