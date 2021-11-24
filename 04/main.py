import sys
import math
import numpy as np
import cv2

from matplotlib import pyplot as plt
from classes import Pixel
from classes import Blob

INPUT_IMAGE = "img/60.bmp"
MIN_PIXELS_N = 15               # Minimum blob pixel quantity
FOREGROUND = 255                # Color to fill in Foreground (0 - 255)
BACKGROUND = 0                  # Color to fill in Background (0 - 255)


# Calculates the distance in color spectrum between two pixels for BGR channels.
def calculateDistance(img, seed_x, seed_y, child_x, child_y):
    seed_pixel = img[seed_y][seed_x]
    child_pixel = img[child_y][child_x]

    return abs(int(seed_pixel) - int(child_pixel))

def find_min_max_pixel_position(blob, pixel):
    
        # Mininum X and Y
        if pixel.x < blob.min_x:
            blob.min_x = pixel.x

        if pixel.y < blob.min_y:
            blob.min_y = pixel.y

        # Maximum X and Y
        if pixel.x > blob.max_x:
            blob.max_x = pixel.x

        if pixel.y > blob.max_y:
            blob.max_y = pixel.y
        
        return blob


# Creates an auxiliar 2Dimensional array which will be used for pixel labelling
def createsAuxMatrix(width, height):
    row = []
    cols = []

    for y in range(height):
        for x in range(width):
            p = Pixel(x, y, -1)
            row.append(p)

        cols.append(row)
        row = []

    return cols

# Detects blobs on a given image
def blobDetection(img, min_pixels_n):
    height, width = img.shape[0], img.shape[1]
    stack = []
    list_of_blobs = []
    pixels = createsAuxMatrix(width, height)
    label = 0
    pixels_per_blob = 0
    blob_px_qty = []

    # store most extreme pixels from the image in axis x and y
    global_min_max_pos = {'min': {'x': width-1, 'y': height-1 }, 
                        'max': {'x': 0, 'y': 0}}

    for y in range(height):
        for x in range(width):

            pix = img[y,x]
            # Does this pixel was never visited before AND is this a foreground? If yes, then we found a seed pixel of a blob
            if pixels[y][x].label == -1 and pix == FOREGROUND:
                pixels[y][x].label = label

                stack.append(pixels[y][x]) # Seed pixel in put inside a stack
                b = Blob(width, height)

                # While stack is not empty, verify neighbors from pixel which was never visited before
                while len(stack) != 0:
                    p = stack.pop()

                    # Add it to blob's pixel list
                    b.pixels_list.append(p)
                    pixels_per_blob += 1
                    # checks if this new pixel is a new min/max position in that blob 
                    b = find_min_max_pixel_position(b, p)

                    # Neighboard 4
                    # Top
                    if p.y - 1 >= 0:
                        if pixels[p.y-1][p.x].label == -1:
                            dist = calculateDistance(img, p.x, p.y, p.x, p.y-1)
                            if dist == 0:
                                pixels[p.y-1][p.x].label = label
                                stack.append(pixels[p.y-1][p.x])
                    # Right
                    if p.x + 1 < width:
                        if pixels[p.y][p.x+1].label == -1:
                            dist = calculateDistance(img, p.x, p.y, p.x+1, p.y)
                            if dist == 0:
                                pixels[p.y][p.x+1].label = label
                                stack.append(pixels[p.y][p.x+1])
                    # Bottom
                    if p.y + 1 < height:
                        if pixels[p.y+1][p.x].label == -1:
                            dist = calculateDistance(img, p.x, p.y, p.x, p.y+1)
                            if dist == 0:
                                pixels[p.y+1][p.x].label = label
                                stack.append(pixels[p.y+1][p.x])
                    # Left
                    if p.x - 1 >= 0:
                        if pixels[p.y][p.x-1].label == -1:
                            dist = calculateDistance(img, p.x, p.y, p.x-1, p.y)
                            if dist == 0:
                                pixels[p.y][p.x-1].label = label
                                stack.append(pixels[p.y][p.x-1])


                # If there is no more pixels in the stack, then all pixels from this blob were found
                b.pixels_qty = pixels_per_blob


                # If blob doesn't have minimum qty of pixels, ignore it
                if b.pixels_qty >= min_pixels_n:
                    list_of_blobs.append(b)
                    blob_px_qty.append(pixels_per_blob)

                    # checks if within this blob, there is a new min/max global position
                    if global_min_max_pos['min']['x'] > b.min_x:
                        global_min_max_pos['min']['x'] = b.min_x
                    if global_min_max_pos['min']['y'] > b.min_y:
                        global_min_max_pos['min']['y'] = b.min_y
                    if global_min_max_pos['max']['x'] < b.max_x:
                        global_min_max_pos['max']['x'] = b.max_x
                    if global_min_max_pos['max']['y'] < b.max_y:
                        global_min_max_pos['max']['y'] = b.max_y
                else:
                    label -= 1 #returns label

                # Reset values for next iteration
                pixels_per_blob = 0
                label += 1


    return global_min_max_pos, blob_px_qty, list_of_blobs

def update_blob_list(blob_qty_px, blob_list):
    mean = np.mean(blob_qty_px)
    stdev = np.std(blob_qty_px)

    print(mean, stdev)

    # Percorre toda lista de blobs
    for k in range(len(blob_list)):
        # Calcula Z-Score desse blob
        zscore = math.sqrt(pow(blob_list[k].pixels_qty - mean, 2))/stdev
        if zscore > 2.5:
            print("cheguei aqui")
            # Quantity of times which this blob needs to be divided (cada vez que passa do zscore 3, é um novo blob)
            qty_divisions = math.ceil(3/zscore)

            for j in range(qty_divisions):
                new_blob_px_qty = math.floor(blob_list[k].pixels_qty/qty_divisions+1)
                # Pega valor de label, aumenta +1
                # Pega aleatoriamente metade dos pixels e cria novo blob pra eles
                blob_list[k].pixels_qty = new_blob_px_qty
                new_blob = Blob()
                new_blob.pixels_qty = new_blob_px_qty
                blob_list.append(new_blob)

    return blob_list
            
def image_resize(img, global_min_max_pos):
    original_height = img.shape[0]
    original_width = img.shape[1]

    img_crop = img[global_min_max_pos['min']['y']:global_min_max_pos['max']['y'], global_min_max_pos['min']['x']: global_min_max_pos['max']['x']]
    diff_height = (original_height - img_crop.shape[0])/original_height
    diff_width = (original_width - img_crop.shape[1])/original_width

    # Verifies if width or height from the cropped image will be used to adjust the image to its new proportion
    if diff_width <= diff_height:
        new_proportion = diff_width
    else:
        new_proportion = diff_height

    print(diff_width, diff_height, new_proportion, img_crop.shape[1], img_crop.shape[0])

    img_resize = cv2.resize(img_crop, (int(img_crop.shape[1] + original_width * new_proportion), int(img_crop.shape[0] + original_height * new_proportion)), interpolation=cv2.INTER_LINEAR)

    return img_resize




def main ():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    
    # Apply image normalization
    aux_norm = np.zeros((800,800))
    img_norm = cv2.normalize(img, aux_norm, 0, 255, cv2.NORM_MINMAX)

    # Apply meadian blur
    img_blur = cv2.medianBlur(img_norm,5)

    # Adaptive thresholding
    img_final = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,-30)

    # Change image type
    #img_final = img_final.astype (np.float32) / 255

    # Image segmentation
    global_min_max_pos, blob_px_qty, list_of_blobs = blobDetection (img_final, MIN_PIXELS_N)

    # Crop region of interest and resize it within the image (it will be used to adjust all images to have the same ratio)
    img_resize = image_resize(img_final.copy(), global_min_max_pos)

    print ('%d componentes detectados. (1)' % len(list_of_blobs))
    #list_of_blobs = update_blob_list(blob_px_qty, list_of_blobs)
    #print ('%d componentes detectados. (2)' % len(list_of_blobs))

    cv2.imshow('01 - out', img_final)
    cv2.imshow('02 - out', img_resize)
    #cv2.imwrite('02 - out.png', img_final*255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main ()

#===============================================================================
