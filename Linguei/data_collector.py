# LINGUEI
# AUTORES:
        # Pedro Cortés
        # Daniel Marques

# IMPORTAR LIBRERIAS A UTILIZAR
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp

# IINICIALIZAMOS LA CAMARA CON OPENCV
cap = cv2.VideoCapture(0)

# INICIALIZAMOS NUESTRO DETECTOR DE MANOS
# EN ESTE CASO IGUALAMOS A 1 YA QUE SOLO SE ESTARA UTILIZANDO 1 MANO
detector = HandDetector(maxHands=1)

# VARIABLES GLOBALES PARA DEFINIR NUESTROS TAMAÑOS PARA LA IMAGEN
offset = 20
imgSize = 300

# CONEXION A CARPETA DONDE GUARDAREMOS NUESTRAS IMAGENES CAPTURADAS
folder = "Data/HOLA"
# CONTADOR DE IMAGENES
counter = 0

while True:
    success, img = cap.read() # INICIAMOS LECTURA DE IMAGENES
    hands, img = detector.findHands(img) # INICIMAOS EL DETECTOR
    if hands:
        hand = hands[0]
        # DEFINIMOS LAS VARIBALES (X,Y,W,H) DENTRO DE UN BOUNDING BOX PARA RECORTAR LA IMAGEN
        x, y, w, h = hand['bbox'] #x y widht and height / bounding box to crop image

        # CREAMOS NUESTRA IMAGEN CON FONDO BLANCO PARA TENER TODAS NUESTRAS MUESTRAS DEL MISMO TAMAÑO (NUMPY MATRIX)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # IMAGEN RECORTADA UNICAMENTE DE LA MANO
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # COLOCAMOS NUESTRA IMAGEN RECORTADA DENTO DE NUESTRO CUADRO BLANCO
        imgCropShape = imgCrop.shape

        # CREAMOS UNA VARIABLE PARA QUE NUESTRA IMAGEN RECORTADA QUEPA DENTRO DEL CUADRADO BLANCO
        aspectRatio = h / w

        # SI ALTURA ES MAS GRANDE QUE EL ANCHO
        if aspectRatio > 1:
            #k = CONSTANTE (TAMAÑO DE IMAGEN / ALTURA)
            k = imgSize / h
            # wCal = ANCHURA CALCULADA (CEIL NOS REDONDEA HACIA ARRIBA)
            wCal = math.ceil(k * w)
            #  HACEMOS EL RESIZE DE NUESTRA IMAGEN (wCal,300)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            # ALTURA SIEMPRE 300
            imgResizeShape = imgResize.shape
            # wGap = BRECHA EN LA ANCHURA - ENCONTRAMOS AL BRECHA PARA CENTRAR LA IMAGEN
            wGap = math.ceil((imgSize - wCal) / 2)
            # CENTRAMOS LA IMAGEN
            imgWhite[:, wGap:wCal + wGap] = imgResize

        # SI ANCHURA ES MAS GRANDE QUE EL ALTO (MISMO CODIGO QUE EN EL IF ANTERIOR PERO CAMBIADO PARA LA ANCHURA)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # NOS MUESTRA LA IMAGEN RECORTADA SIN EL RESIZE EN EL CUADRO BALNCO
        cv2.imshow("ImageCrop", imgCrop)
        # RESIZE DE IMAGEN EN CUADRO BLANCO
        cv2.imshow("ImageWhite", imgWhite)

    # MOSTRAMOS NUESTRA IMAGEN PRINCIPAL
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)


    # AL PRESIONAR LA TECLA "S" GUARDAMOS LA IMAGEN
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)