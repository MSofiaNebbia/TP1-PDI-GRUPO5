#PROBLEMA1 TP1 PDI GRUPO 5

import cv2
import numpy as np
import matplotlib.pyplot as plt

#ECUALIZACIÓN LOCAL DE HISTOGRAMA. 

def ecualizacion_local_histograma(imagen, M, N):
    filas, columnas = imagen.shape

    margen_y = M // 2
    margen_x = N // 2

    #extendemos para poder trabajar con los extremos tambien
    imagen_extendida = cv2.copyMakeBorder(imagen, margen_y, margen_y, margen_x, margen_x, cv2.BORDER_REPLICATE)

    imagen_resultado = np.zeros_like(imagen, dtype = np.uint8)

    #recorremos cada pixel de imagen original
    for i in range(filas):
        for j in range(columnas):

            #region local de pixel actual (vecindario)
            region = imagen_extendida[i:i+M, j:j+N]

            #calculamos histograma de esa region 
            histograma, _ = np.histogram(region, bins=256, range=(0, 256))

            #normalizamos 
            histograma = histograma / (M * N)

            suma_acumulada = np.cumsum(histograma)

            #nuevo valor pixel central
            pixel_central = imagen_extendida[i + margen_y, j + margen_x]
            nuevo_valor = 255 * suma_acumulada[pixel_central]

            imagen_resultado[i, j] = np.uint8(nuevo_valor)

    return imagen_resultado

imagen = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

resultado_3x3 = ecualizacion_local_histograma(imagen, 3, 3)
plt.figure()
plt.imshow(resultado_3x3, cmap='gray')
plt.show()