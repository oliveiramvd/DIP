import numpy as np
import cv2
from matplotlib import pyplot as plt
#===============================================================================

INPUT_IMAGE =  'img/114.bmp'

# TODO: ajuste estes parâmetros!
ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 5

#===============================================================================

def img_auxiliar(img, height, width):
    return np.full((height, width), -1)

def verifica_altura_largura(componente, altura_min, largura_min):
    ok_altura = False
    ok_largura = False

    if componente['B'] - componente['T'] >= altura_min or componente['B'] - componente['T'] == 0:
        ok_altura = True
    if componente['R'] - componente['L'] >= largura_min or componente['R'] - componente['L'] == 0:
        ok_largura = True
    
    if ok_altura and ok_largura:
        return True
    else:
        return False

def atualiza_coordenadas(componente, y, x):
    if componente['L'] > x:
        componente['L'] = x
    if componente['B'] < y:
        componente['B'] = y
    if componente['R'] < x:
        componente['R'] = x

def rotula (img, largura_min, altura_min, n_pixels_min):
    height = img.shape[0]
    width = img.shape[1]
    img_aux = img_auxiliar(img, height, width)
    label = 0
    lista_componentes = []
    
    # Para cada pixel da imagem...
    for y in range(height):
        for x in range(width):
            # Pixel é um foreground e não foi marcado com label ainda
            if img[y, x] == 1 and img_aux[y, x] == -1:
                componente = {}
                componente['label'] = label
                componente['n_pixels'] = 1
                # Inicialmente o tamanho total do componente é do pixel semente
                componente['T'], componente['B'] = y, y    
                componente['L'], componente['R'] = x, x
                inunda(label, img, img_aux, x, y, componente)

                # Valida se componente possui altura, largura e qtd mínima de pixels
                if componente['n_pixels'] >= n_pixels_min and verifica_altura_largura(componente, altura_min, largura_min):
                    lista_componentes.append(componente)
                    label += 1

    return lista_componentes
    
def inunda(label, img, img_aux, x, y, componente):
    height = img.shape[0]
    width = img.shape[1]
    img_aux[y, x] = label

    # Vizinhança 4
    # Acima
    if y-1 >= 0:
        if img[y-1, x] == 1 and img_aux[y-1, x] == -1:
            componente['n_pixels'] += 1
            atualiza_coordenadas(componente, y-1, x)
            inunda(label, img, img_aux, x, y-1, componente)
    # Esquerda
    if x-1 >= 0:
        if img[y, x-1] == 1 and img_aux[y, x-1] == -1:
            componente['n_pixels'] += 1
            atualiza_coordenadas(componente, y, x-1)
            inunda(label, img, img_aux, x-1, y, componente)
    # Baixo
    if y+1 < height:
        if img[y+1, x] == 1 and img_aux[y+1, x] == -1:
            componente['n_pixels'] += 1
            atualiza_coordenadas(componente, y+1, x)
            inunda(label, img, img_aux, x, y+1, componente)
    # Direita
    if x+1 < width:
        if img[y, x+1] == 1 and img_aux[y, x+1] == -1:
            componente['n_pixels'] += 1
            atualiza_coordenadas(componente, y, x+1)
            inunda(label, img, img_aux, x+1, y, componente)

def main ():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    
    # Normaliza usando min/max
    aux_norm = np.zeros((800,800))
    img_norm = cv2.normalize(img, aux_norm, 0, 255, cv2.NORM_MINMAX)

    img_blur = cv2.medianBlur(img_norm,5)
    #img_blur = cv2.bilateralFilter(img_norm,9,75,75)

    th1 = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,-30)
    # aplica blur
    

  

    #Limiarização adaptativa

    titles = ['Original Image', 'Norm', 'Adaptive']
    images = [img, img_norm, th1]
    for i in range(3):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



if __name__ == '__main__':
    main ()

#===============================================================================
