import sys
import math
import timeit
import cv2
import numpy as np

INPUT_IMAGE =  'img/02.jpg'
WINDOW_HEIGHT = 3
WINDOW_WIDTH = 3


# Applies a mean Filter in an image using a simple algorithm
def mean_filter(img_in, w_height, w_width):
    height = img_in.shape[0]
    width = img_in.shape[1]
    img_out = img_in.copy()

    # Goes through every pixel within the image
    for y in range(height):
        for x in range(width):
            sum = np.zeros(3)
            # Slides through the image applying a arithmetic mean. Also, verifies if window don't exceed the image size (if so, sums only the pixels which doesn't exceed the image size for that window)
            for j in range(y - math.floor(w_height/2), y + math.floor(w_height/2)+1):
                for i in range(x - math.floor(w_width/2), x + math.floor(w_width/2)+1):
                    if i >= 0 and i < width and j >= 0 and j < height:
                        sum += img_in[j, i]
                
            img_out[y, x] = sum/(w_height * w_width)
    
    return img_out


# Applies a mean Filter using a somewhat enhanced algorithm than the previous one (first, it blurs the image horizontally and then vertically)
def mean_filter_2pass(img_in, w_height, w_width):
    height = img_in.shape[0]
    width = img_in.shape[1]
    img_buffer = img_in.copy()

    # First pass (Horizontally)
    for y in range(height):
        for x in range(width):
            sum = np.zeros(3)
            # Slides through the image applying a arithmetic mean. Also, verifies if window don't exceed the image size (if so, sums only the pixels which doesn't exceed the image size for that window)
            for i in range(x - math.floor(w_width/2), x + math.floor(w_width/2)+1):
                if i >= 0 and i < width:
                    sum+= img_in[y, i]
                
                img_buffer[y, x] = sum/w_width
    
    img_out = img_buffer.copy()

    # Second pass (Vertically)
    for x in range(width):
        for y in range(height):
            sum = np.zeros(3)
            for j in range(y - math.floor(w_height/2), y + math.floor(w_height/2)+1):
                if j >= 0 and j < height:
                    sum += img_buffer[j, x]
                
                img_out[y, x] = sum/w_height
    
    return img_out


# Creates an integral image which will be used to apply a mean filter
def create_integral_image(img_in):
    height = img_in.shape[0]
    width = img_in.shape[1]
    int_img = img_in.copy()

    # Sums pixels horizontally
    for y in range(height):
        for x in range(width):
            if x == 0:
                int_img[y, x] = int_img[y, x]
            else:
                int_img[y, x] +=  int_img[y, x-1]

    # Sums pixels vertically
    for x in range(width):
        for y in range(height):
            if y != 0:
                int_img[y, x] += int_img[y-1, x]

    return int_img

# Applies a mean Filter using an integral image
def mean_filter_integral(img_in, int_img, w_height, w_width):
    height = img_in.shape[0]
    width = img_in.shape[1]
    img_out = img_in.copy()

    for y in range(height):
        for x in range(width):
            if y-1 - math.floor(w_height/2) >= 0 and x-1 - math.floor(w_width/2) >= 0 and y+1 + math.floor(w_height/2) < height and x+1 + math.floor(w_width/2) < width:

                img_out[y, x] = (int_img[y + math.floor(w_height/2), x + math.floor(w_width/2)] - int_img[y-1 - math.floor(w_height/2), x + math.floor(w_width/2)] - int_img[y + math.floor(w_height/2), x-1 - math.floor(w_width/2)] + int_img[y-1 - math.floor(w_height/2), x-1 - math.floor(w_width/2)]) / (w_width * w_height)

    return img_out

def main ():

    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Error while opening image.\n')
        sys.exit ()

    img = img.astype (np.float32) / 255

    start_time = timeit.default_timer ()
    img_out = mean_filter(img, WINDOW_HEIGHT, WINDOW_WIDTH)
    print ('Elapsed time (Mean Filter 1): %f' % (timeit.default_timer () - start_time))
    
    start_time = timeit.default_timer ()
    img_out2 = mean_filter_2pass(img, WINDOW_HEIGHT, WINDOW_WIDTH)
    print ('Elapsed time (Mean Filter 2): %f' % (timeit.default_timer () - start_time))

    start_time = timeit.default_timer ()
    int_img = create_integral_image(img)
    img_out3 = mean_filter_integral(img, int_img, WINDOW_HEIGHT, WINDOW_WIDTH)
    print ('Elapsed time (Mean Filter 3): %f' % (timeit.default_timer () - start_time))

    cv2.imwrite ('img/01 - Mean Filter.png', img_out*255)
    cv2.imwrite ('img/02 - Mean Filter - Two-pass.png', img_out2*255)
    cv2.imwrite ('img/03 - Mean Filter - Integral.png', img_out3*255)


if __name__ == '__main__':
    main ()
