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

imagen = cv2.imread("Imagen_con_detalles_escondidos.tif", cv2.IMREAD_GRAYSCALE)

resultado_3x3 = ecualizacion_local_histograma(imagen, 3, 3)
plt.figure(), plt.imshow(resultado_3x3, cmap="gray"), plt.title("Ecualizacion local 3 x 3"), plt.show(block=False)

resultado_9x9=ecualizacion_local_histograma(imagen,9,9)
plt.figure(), plt.imshow(resultado_9x9, cmap="gray"),plt.title("Ecualizacion local 9 x 9"), plt.show(block=False)

resultado_25x25 = ecualizacion_local_histograma (imagen, 25, 25)
plt.figure(), plt.imshow(resultado_25x25, cmap="gray"),plt.title("Ecualizacion local 25 x 25"), plt.show(block=False)

plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.title("Imagen original")
plt.imshow(imagen, cmap="gray")

plt.subplot(2,2,2)
plt.title("Ecualización local - 3x3")
plt.imshow(resultado_3x3, cmap="gray")

plt.subplot(2,2,3)
plt.title("Ecualización local - 9x9")
plt.imshow(resultado_9x9, cmap="gray")

plt.subplot(2,2,4)
plt.title("Ecualización local - 25x25")
plt.imshow(resultado_25x25, cmap="gray")
plt.show()