import sys
import math
import timeit
import cv2
import numpy as np

INPUT_IMAGE =  'img/bleach.jpg'
NEGATIVO = True
WINDOW_SIZE = 25

def mean_blur(img_in, w_size):
    height = img_in.shape[0]
    width = img_in.shape[1]
    #channels = img_in.shape[2]
    img_out = img_in.copy()

    for y in range(height):
        for x in range(width):
            # Verify if window don't exceed the image size
            if y - math.floor(w_size/2) >= 0 and x - math.floor(w_size/2) >= 0 and y + math.floor(w_size/2) <= height and x + math.floor(w_size/2) <= width:
                img_out[y, x][0] = 1
                img_out[y, x][1] = 1
                img_out[y, x][2] = 1
            else:
                img_out[y, x][0] = 0
                img_out[y, x][1] = 0
                img_out[y, x][2] = 0
    
    return img_out


def main ():

    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Error while opening image.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.astype (np.float32) / 255

    start_time = timeit.default_timer ()
    print ('Elapsed time: %f' % (timeit.default_timer () - start_time))
    img_out = mean_blur(img, WINDOW_SIZE)
    cv2.imshow ('01 - binarizada', img_out)
    cv2.imwrite ('01 - binarizada.png', img_out*255)
    #mean_blur_2pass
    #mean_blur_integral

    # Mostra os objetos encontrados.
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()
