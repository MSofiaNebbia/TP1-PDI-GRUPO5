#PROBLEMA2 TP1 PDI GRUPO 5

#IMPORTACION DE LIBRERIAS 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

#RESPUESTAS CORRECTAS 
RESPUESTAS_CORRECTAS = {1: 'C', 2: 'B', 3: 'A', 4: 'D', 5: 'B',
6: 'B', 7: 'A', 8: 'B', 9: 'D', 10: 'D'}

#FUNCIONES AUXILIARES

def imshow(img, titulo="Imagen"):
    "Muestra una imagen en pantalla con Matplotlib"
    plt.figure(figsize=(10, 6))

    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    else:
        plt.imshow(img, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def compactar(valores, distancia_min):
    "Elimina valores muy cercanos entre si de una lista"
    if len(valores) == 0:  #si la lista esta vacia retorna vacío
        return []
    
    resultado = [valores[0]] #1er valor

    for v in valores[1:]:
        if v - resultado[-1] > distancia_min: #si esta lo suficientemente lejos
            resultado.append(v)

    return resultado #lista sin valores muy juntos

def contar_componentes(celda, area_min=20):
    "Cuenta componentes conectada en una imagen binaria filtrando por area minima"
    if celda.size == 0:
        return 0, [] #0 componentes
    
    num, _, stats, _ = cv2.connectedComponentsWithStats(celda, connectivity=8) #detecta grupos de píxeles blancos conectados
    cajas = [stats[i] for i in range(1, num) #recorro desde 1 xq 0 es fondo
    if stats[i, cv2.CC_STAT_AREA] > area_min] #filtro por area min
    return len(cajas), cajas 

def hay_espacio_entre_palabras(cajas, factor=2.0):
    "Detecta si hay un espacio grande entre dos grupos de letras (dos palabras)"
    if len(cajas) < 2: #necesitamos al menos 2 componentes
        return False
    
    cajas_ord = sorted(cajas, key=lambda c: c[cv2.CC_STAT_LEFT])
    separacion = []

    for i in range(1, len(cajas_ord)):
        x_fin = cajas_ord[i-1][cv2.CC_STAT_LEFT] + cajas_ord[i-1][cv2.CC_STAT_WIDTH] #x donde termina componente anterior
        x_sig = cajas_ord[i][cv2.CC_STAT_LEFT] #x donde empieza componente siguiente
        separacion.append(x_sig - x_fin) #separacion entre componentes

    separacion_positiva = [g for g in separacion if g > 0] #seps reales

    if len(separacion_positiva) == 0:
        return False

    sep_promedio = np.mean(separacion_positiva)
    sep_max = np.max(separacion_positiva) #posible espacio entre palabras
    idx_max = np.argmax(separacion) #indice separacion mas grande

    componentes_antes = idx_max + 1 #cantidad de comp a la izq de la sep
    componentes_despues = len(cajas_ord) - componentes_antes #a la der
    tiene_dos_palabras = componentes_antes >= 2 and componentes_despues >= 3  #min 2 letras x palabra

    return sep_max >= 5 and sep_max > factor * sep_promedio and tiene_dos_palabras
    #la separacion maxima tiene que ser grande y mayor al promedio

def cargar_y_binarizar(path):
    "Carga una imagen en escala de grises y la binariza con umbral adaptativo"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bin_inv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    #binarización adaptativa gaussiana

    return img, bin_inv

def detectar_lineas(bin_inv):
    "Detecta lineas horizontales y verticales de la grilla usando morfologia"
    alto, ancho = bin_inv.shape #dim img
    kernel_h = np.ones((1, ancho // 10), dtype=np.uint8)  #1fila, muchas columnas
    #detecta elementos que se extiendan horizontalmente

    kernel_v = np.ones((alto // 10, 1), dtype=np.uint8)  
    #detecta lineas verticales largas 

    lineas_h = cv2.erode(bin_inv, kernel_h)   #erosion horizontal
    lineas_h = cv2.dilate(lineas_h, kernel_h) #dilatacion horizontal

    lineas_v = cv2.erode(bin_inv, kernel_v)    #erosion vertical
    lineas_v = cv2.dilate(lineas_v, kernel_v)  #dilatacion vertical

    return lineas_h, lineas_v

def obtener_coords_lineas(lineas_h, lineas_v):
    "Obtiene las coordenadas (y) de las filas y (x) de las columnas"
    filas = np.where(np.sum(lineas_h, axis=1) > lineas_h.shape[1] * 0.3)[0]
    #sumamos píxeles por fila si supera el 30% del ancho hay una línea horizontal

    filas = compactar(filas, distancia_min=15)
    #eliminamos filas muy cercanas (líneas de más de 1 píxel de ancho)

    cols  = np.where(np.sum(lineas_v, axis=0) > lineas_v.shape[0] * 0.2)[0]
    #sumamos píxeles por columna si supera 20% del alto hay una línea vertical

    cols  = compactar(cols,  distancia_min=15)
    return filas, cols #coordenadas de filas y columnas de la grilla

def detectar_campos_encabezado(img_gris, filas, umbral_px=150, racha_min=30, margen=3):
    "Detecta los 3 campos del encabezado buscando pixeles oscuros en la fila del subrayado"
    f_sub = filas[0]
    fila = img_gris[f_sub, :]

    rachas = []
    racha = 0
    x_ini = 0
    for x, px in enumerate(fila):
        if px < umbral_px:
            if racha == 0:
                x_ini = x
            racha += 1
        else:
            if racha >= racha_min:
                rachas.append((x_ini, x - 1))
            racha = 0
    if racha >= racha_min:
        rachas.append((x_ini, len(fila) - 1))

    #si no detecta exactamente 3 campos algo salio mal
    if len(rachas) != 3:
        print(f"Error: se esperaban 3 campos y se encontraron {len(rachas)}")
        return None

    y1 = max(0, f_sub - 20)
    y2 = f_sub

    return {
        'name':  (y1, y2, rachas[0][0] - margen, rachas[0][1] + margen),
        'date':  (y1, y2, rachas[1][0] - margen, rachas[1][1] + margen),
        'class': (y1, y2, rachas[2][0] - margen, rachas[2][1] + margen),
    }

#VALIDACION DE LOS ENCABEZADOS 
def validar_name(celda_bin):
    n, cajas = contar_componentes(celda_bin, area_min=15)
    if n < 3:
        return 'MAL'
    if n > 25:                          # verifica maximo 25 caracteres
        return 'MAL'
    if hay_espacio_entre_palabras(cajas, factor=2.0): #verifica al menos 2 palabras
        return 'OK'
    return 'MAL'

def validar_date(celda_bin):
    n, cajas = contar_componentes(celda_bin, area_min=15)
    
    if n != 8: #verifica exactamente 8 caracteres
        return 'MAL'
    
    if hay_espacio_entre_palabras(cajas, factor=2.0): #verifica ser una sola palabra
        return 'MAL'
    
    return 'OK'

def validar_class(celda_bin):
    n, _ = contar_componentes(celda_bin, area_min=15)
    return 'OK' if n == 1 else 'MAL' #verifico exactamente un caracter

#FUNCIONES PRINCIPALES

def encontrar_linea_y_recortar(celda_bin):
    "Busca la linea horizontal de respuesta dentro de una celda y devuelve la franja que la contiene"
    alto  = celda_bin.shape[0]
    ancho = celda_bin.shape[1]

    RACHA_MIN = 30  #la linea tiene al menos 30 pixeles blancos consecutivos

    fila_linea = None

    for y in range(3, alto - 3): #recorre las filas evitando los bordes
        fila = celda_bin[y, :]
        racha_max = 0
        racha_actual = 0
        for px in fila:
            if px > 0: #pixel es blanco
                racha_actual += 1
                if racha_actual > racha_max:
                    racha_max = racha_actual
            else:
                racha_actual = 0
        if racha_max >= RACHA_MIN:
            fila_linea = y
            break  #primera fila que cumple = linea 

    if fila_linea is None:
        return None

    alto_letra = 15
    y1 = max(0, fila_linea - alto_letra)
    y2 = fila_linea + 2  #+2 para incluir el borde superior de la linea

    return celda_bin[y1:y2, :]

def recortar_letra_sobre_linea(recorte_bin):
    "Recorta exactamente el ancho de la linea de respuesta y los pixeles encima donde esta escrita la letra"
    alto  = recorte_bin.shape[0]
    ancho = recorte_bin.shape[1]

    RACHA_MIN = 30

    #buscamos la fila de la linea de abajo hacia arriba
    fila_linea = None
    x_inicio_linea = None
    x_fin_linea    = None

    for y in range(alto - 1, 0, -1): #empieza en la ult fila
        fila = recorte_bin[y, :] #extraigo pix fila actual
        racha_actual = 0
        racha_max    = 0
        x_ini_actual = 0
        x_ini_mejor  = 0
        x_fin_mejor  = 0

        for xi, px in enumerate(fila):
            if px > 0:
                if racha_actual == 0:
                    x_ini_actual = xi
                racha_actual += 1
                if racha_actual > racha_max:
                    racha_max   = racha_actual
                    x_ini_mejor = x_ini_actual
                    x_fin_mejor = xi
            else:
                racha_actual = 0

        if racha_max >= RACHA_MIN:
            fila_linea     = y
            x_inicio_linea = x_ini_mejor
            x_fin_linea    = x_fin_mejor
            break

    if fila_linea is None:
        return None

    #recortamos exactamente el ancho de la linea y encima
    y2 = fila_linea
    y1 = max(0, y2 - 15)

    return recorte_bin[y1:y2, x_inicio_linea:x_fin_linea]

def identificar_letra(zona_bin):
    "Identifica la letra escrita (A, B, C o D) analizando huecos internos y proyeccion vertical"
    if zona_bin.size == 0:
        return None

    n, cajas = contar_componentes(zona_bin, area_min=8)
    if n == 0:
        return None #no hay nada escrito

    #tomamos la componente más grande (es la letra)
    caja = max(cajas, key=lambda c: c[cv2.CC_STAT_AREA])

    x, y = caja[cv2.CC_STAT_LEFT], caja[cv2.CC_STAT_TOP]
    w, h = caja[cv2.CC_STAT_WIDTH], caja[cv2.CC_STAT_HEIGHT]

    if h < 4:
        return None

    margen = 1
    recorte = zona_bin[max(0, y-margen) : y+h+margen,
                    max(0, x-margen) : x+w+margen]

    if recorte.size == 0:
        return None

    #contamos huecos internos
    contornos, jerarquia = cv2.findContours(recorte, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    n_hijos = 0
    if jerarquia is not None:
        for i, j in enumerate(jerarquia[0]):
            if j[3] != -1:
                if cv2.contourArea(contornos[i]) > 3:
                    n_hijos += 1

    if n_hijos == 0:
        return 'C'
    if n_hijos >= 2:
        return 'B'

    #1 hueco: A o D por proyección vertical
    proy_vertical  = np.sum(recorte == 255, axis=0)
    promedio_total = np.mean(proy_vertical)
    pico_izquierda = np.max(proy_vertical[:3])

    if pico_izquierda >= promedio_total * 2.0:
        return 'D'
    else:
        return 'A'

def hay_multiples_letras(zona_bin):
    "Detecta si hay mas de una letra escrita en la linea (respuesta invalida)"
    if zona_bin is None or zona_bin.size == 0: #no se encontro linea o esta vacia
        return False

    n, cajas = contar_componentes(zona_bin, area_min=8)
    if n == 0:
        return False

    #filtramos componentes muy chicas (ruido, punto)
    cajas_letras = [c for c in cajas
                    if c[cv2.CC_STAT_HEIGHT] >= 6
                    and c[cv2.CC_STAT_AREA] >= 20]

    return len(cajas_letras) >= 2

def generar_imagen_resultados(resultados):
    "Genera una imagen con el crop del nombre de cada examen, con borde verde si aprobo y rojo si desaprobo"
    alto_celda = 50
    ancho_celda = 320
    margen = 10

    alto_total = (alto_celda + margen) * len(resultados) + margen
    ancho_total = ancho_celda + margen * 2

    imagen = np.ones((alto_total, ancho_total, 3), dtype=np.uint8) * 255

    for idx, (nombre, crop, puntaje, aprobado) in enumerate(resultados):
        y = margen + idx * (alto_celda + margen)
        x = margen

        color = (0, 180, 0) if aprobado else (0, 0, 200)

        crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        crop_resized = cv2.resize(crop_color, (ancho_celda, alto_celda))
        imagen[y:y+alto_celda, x:x+ancho_celda] = crop_resized
        grosor = 4
        cv2.rectangle(imagen, (x - grosor, y - grosor),
                      (x + ancho_celda + grosor, y + alto_celda + grosor), color, grosor)

    return imagen

# PROGRAMA PRINCIPAL    
def procesar_examen(path):
    "Procesa un examen completo: valida el encabezado, detecta y corrige las 10 respuestas"
    img_gris, bin_inv = cargar_y_binarizar(path)
    lineas_h, lineas_v = detectar_lineas(bin_inv)
    filas, cols = obtener_coords_lineas(lineas_h, lineas_v)

    campos = detectar_campos_encabezado(img_gris, filas)

    if campos is None:   #si no encontro los 3 campos
        print("Error: no se pudieron detectar los campos del encabezado")
        return None, None, None

    #desempaquetamos coordenadas por campo
    y1n, y2n, x1n, x2n = campos['name']
    y1d, y2d, x1d, x2d = campos['date']
    y1c, y2c, x1c, x2c = campos['class']

    celda_name  = bin_inv[y1n:y2n, x1n:x2n]
    celda_date  = bin_inv[y1d:y2d, x1d:x2d]
    celda_class = bin_inv[y1c:y2c, x1c:x2c]

    crop_name = img_gris[max(0, y1n - 5) : y2n + 5, x1n:x2n]

    print(f"\nExamen: {path}")
    print(f"Name:  {validar_name(celda_name)}")
    print(f"Date:  {validar_date(celda_date)}")
    print(f"Class: {validar_class(celda_class)}")

    celdas_enunciados = [
        (filas[1], filas[3],  cols[0], cols[1]),  #P1
        (filas[3], filas[5],  cols[0], cols[1]),  #P2
        (filas[5], filas[7],  cols[0], cols[1]),  #P3
        (filas[7], filas[9],  cols[0], cols[1]),  #P4
        (filas[9], filas[10], cols[0], cols[1]),  #P5
        (filas[1], filas[3],  cols[2], cols[3]),  #P6
        (filas[3], filas[5],  cols[2], cols[3]),  #P7
        (filas[5], filas[7],  cols[2], cols[3]),  #P8
        (filas[7], filas[9],  cols[2], cols[3]),  #P9
        (filas[9], filas[10], cols[2], cols[3]),  #P10
    ]

    correctas = 0
    print("\nRespuestas:")

    for i in range(10):
        f1, f2, c1, c2 = celdas_enunciados[i]
        celda_bin = bin_inv[f1:f2, c1:c2]

        #paso 1, encontrar la linea y recortar la franja con la letra
        zona = encontrar_linea_y_recortar(celda_bin)

        #paso 2, recortar solo la letra encima de la linea
        if zona is not None:
            zona = recortar_letra_sobre_linea(zona)

        #paso 3, verificar si hay múltiples letras (respuesta inválida) y despues identificar la letra
        if zona is None:
            letra = None
        elif hay_multiples_letras(zona):
            #si hay 2 o mas letras escritas es una respuesta invalida
            letra = 'INVALIDA'
        else:
            letra = identificar_letra(zona)

        esperada = RESPUESTAS_CORRECTAS[i+1]
        ok = letra == esperada
        if ok:
            correctas += 1
        print(f"  P{i+1:2d}: detectada={letra}  correcta={esperada}  {'BIEN' if ok else 'MAL'}")

    print(f"\nPuntaje: {correctas}/10")
    aprobado = correctas >= 6
    print(f"Resultado: {'APROBADO' if aprobado else 'DESAPROBADO'}")
    return correctas, aprobado, crop_name

#EJECUCION DEL PROGRAMA
archivos = glob.glob("examen_*.png")  
archivos.sort()                    

resultados = []
for archivo in archivos:
    correctas, aprobado, crop_name = procesar_examen(archivo)  #proces cada examen
    resultados.append((archivo, crop_name, correctas, aprobado))  #guarda los resultados

img_final = generar_imagen_resultados(resultados) 
cv2.imwrite("resultados.png", img_final)           
imshow(img_final, "Resultados")              
