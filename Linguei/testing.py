# LINGUEI
# AUTORES:
        # Pedro Cortés
        # Daniel Marques

# IMPORTAR LIBRERIAS A UTILIZAR
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import mediapipe as mp
import asyncio

# IINICIALIZAMOS LA CAMARA CON OPENCV
cap = cv2.VideoCapture(0)
# INICIALIZAMOS NUESTRO DETECTOR DE MANOS
# EN ESTE CASO IGUALAMOS A 1 YA QUE SOLO SE ESTARA UTILIZANDO 1 MANO
detector = HandDetector(maxHands=1)
# CON CLASSIFIER LEEMOS NUESTRO MODELO DE KERAS Y LAS LABELS CREADAS
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# VARIABLES GLOBALES PARA DEFINIR NUESTROS TAMAÑOS PARA LA IMAGEN
offset = 20
imgSize = 300

# HACEMOS UN LISTADO DE NUESTRAS LETRAS PARA QUE CADA VEZ QUE COINCIDA NUESTRO INDEX NOS MUESTRE LA LETRA/FRASE A PREDECIR
labels = ["A", "E", "I", "O","P","D","R","N","L", "HOLA"]
# LISTA GLOBAL PARA CONCATENAR LETRAS
word = []

# FUNCION PARA CONCATENAR LETRAS Y FORMAR PALABRAS
def concat_letters(list,index):
    list.append(index)
    palabra = "".join(list)
    return palabra

while True:
    success, img = cap.read() # INICIAMOS LECTURA DE IMAGENES
    imgOutput = img.copy()  # REALIZAMOS UNA COPIA DE NUESTRA IMAGEN ORIGINAL PARA NUESTRO OUTPUT
    hands, img = detector.findHands(img) # INICIMAOS EL DETECTOR
    if hands:
        hand = hands[0]
        # DEFINIMOS LAS VARIBALES (X,Y,W,H) DENTRO DE UN BOUNDING BOX PARA RECORTAR LA IMAGEN
        x, y, w, h = hand['bbox']  # x y widht and height / bounding box to crop image

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
            # k = CONSTANTE (TAMAÑO DE IMAGEN / ALTURA)
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
            # FUNCION QUE NOS REGRESA NUESTRA PREDICCION Y NUESTRO INDEX DENTRO DE LABELS
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        # SI ANCHURA ES MAS GRANDE QUE EL ALTO (MISMO CODIGO QUE EN EL IF ANTERIOR PERO CAMBIADO PARA LA ANCHURA)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)


        # DIBUJAMOS UN RECTANGULO ATRAS DE LAS LETRAS PARA VERLAS CORRECTAMENTE
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        # ESCRIBIMOS LA PREDICCION EN PANTALLA MEDIANTE EL INDEX
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)


        # PONEMOS NUESTRA PREDICCION CONCATENADA COMO PALABRA/FRASE
        cv2.putText(imgOutput, concat_letters(word,labels[index]), (100, 600), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)


        #print(word)
        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)

    # MOSTRAMOS NUESTRA IMAGEN PRINCIPAL
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    # AL PRESIONAR LA TECLA C REINICIAMOS NUESTRA VARIABLE WORD PARA INICIAR UNA NUEVA PALABRA
    if key == ord("c"):
        word = []