import sys
import math
import timeit
import cv2
import numpy as np

INPUT_IMAGE =  'img/example.png'
NEGATIVO = True
WINDOW_SIZE = 15

def mean_filter(img_in, w_size):
    height = img_in.shape[0]
    width = img_in.shape[1]
    img_out = img_in.copy()

    for y in range(height):
        for x in range(width):
            # Verify if window don't exceed the image size
            if y - math.floor(w_size/2) >= 0 and x - math.floor(w_size/2) >= 0 and y + math.floor(w_size/2) <= height and x + math.floor(w_size/2) <= width:
                sum = np.zeros(3)
                # Slide through the image applying a mean
                for j in range(y - math.floor(w_size/2), y + math.floor(w_size/2)):
                    for i in range(x - math.floor(w_size/2), x + math.floor(w_size/2)):
                        sum += img_in[j, i]
                
                img_out[y, x] = sum/(w_size * w_size)
    
    return img_out


def mean_filter_2pass(img_in, w_size):
    height = img_in.shape[0]
    width = img_in.shape[1]
    img_out = img_in.copy()

    # First pass (Horizontal)
    for y in range(height):
        for x in range(width):
            # Verify if window don't exceed the image size
            if x - math.floor(w_size/2) >= 0 and x + math.floor(w_size/2) <= height:
                sum = np.zeros(3)
                # Slide through the image applying a mean
                for i in range(x - math.floor(w_size/2), x + math.floor(w_size/2)):
                    sum += img_in[y, i]
                
                img_out[y, x] = sum/(w_size)

    # Second pass (Vertical)
    for y in range(height):
        for x in range(width):
            # Verify if window don't exceed the image size
            if y - math.floor(w_size/2) >= 0 and y + math.floor(w_size/2) <= height:
                sum = np.zeros(3)
                # Slide through the image applying a mean
                for j in range(y - math.floor(w_size/2), y + math.floor(w_size/2)):
                    sum += img_out[j, x]
                
                img_out[y, x] = sum/(w_size)
    


    return img_out

def mean_filter_integral(img_in, w_size):
    return

def main ():

    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Error while opening image.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.astype (np.float32) / 255

    start_time = timeit.default_timer ()
    img_out = mean_filter(img, WINDOW_SIZE)
    print ('Elapsed time (Mean Filter 1): %f' % (timeit.default_timer () - start_time))
    
    start_time = timeit.default_timer ()
    #img_out2 = mean_filter_2pass(img, WINDOW_SIZE)
    print ('Elapsed time (Mean filter 2): %f' % (timeit.default_timer () - start_time))

    img_out3 = cv2.blur(img, (15, 15))

    cv2.imshow ('01 - Mean filter', img_out)
    #cv2.imshow ('02 - Mean filter - Two-pass', img_out2)
    cv2.imshow ('03 - Mean filter - OpenCV', img_out3)
    cv2.imwrite ('img/01 - binarizada.png', img_out*255)

  
    
    #mean_blur_integral

    # Mostra os objetos encontrados.
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()
