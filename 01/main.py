#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2
from numpy.lib.type_check import imag

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 10

#===============================================================================

def binariza (img, threshold):
    #binarizar cada canal independentemente

    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!
    img_out = img
    height, width, channel = img_out.shape

    for y in range(height):
        for x in range(width):
            pixel = img_out[y, x]

            # BGR ou grayscale?
            if channel == 3:
                px = (0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2])
            else:
                px = pixel[0]

            # Dentro do limiar?
            if px >  threshold:
                img_out[y, x] = 1
            else:
                img_out[y, x] = 0
    
    return img_out


# Matriz auxiliar informando o label para cada posição f(x, y)
def img_auxiliar(img, height, width):
    return np.zeros((height, width), dtype=float)

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    height = img.shape[0]
    width = img.shape[1]
    img_aux = img_auxiliar(img, height, width)
    label = 0.1
    componente = {}
    lista_componentes = [{}]
    
    # Para cada pixel da imagem...
    for y in range(height):
        for x in range(width):
            # Pixel é um foreground e não foi marcado com label ainda
            if img[y, x] == 1 and img_aux[y, x] == 0:
                componente = inunda(label, img, img_aux, x, y, componente)
                label += 0.1
    return lista_componentes

    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    
def inunda(label, img, img_aux, x, y, componente):
    img_aux[y, x] = label
    componente['label'] = label
    #componente['n_pixels'] = 1
    #componente['coordenadas'] = {'T': 0, 'L':0, 'B':0, 'R':0}

    # Vizinhança 4
    # Acima
    if y-1 >= 0:
        if img[y-1, x] == 1 and img_aux[y-1, x] == 0:
            img_aux[y-1, x] = label
            inunda(label, img, img_aux, x, y-1, componente)

    # Esquerda

    # Baixo

    # Direita

    inunda()

    return 0


#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('teste',img)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()

    lista_componentes = [{'1':0,'2':0}]
    print(lista_componentes)
    #componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    #n_componentes = len (componentes)
    #print ('Tempo: %f' % (timeit.default_timer () - start_time))
    #print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    #for c in componentes:
    #    cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    #cv2.imshow ('02 - out', img_out)
    #cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
