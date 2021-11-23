import sys
import math
import timeit
import numpy as np
import cv2

from matplotlib import pyplot as plt
from classes import Pixel
from classes import Blob


INPUT_IMAGE = "img/82.bmp"
MIN_PIXELS_N = 15               # Minimum blob pixel quantity
FOREGROUND = 255                # Color to fill in Foreground (0 - 255)
BACKGROUND = 0                  # Color to fill in Background (0 - 255)


# Calculates the distance in color spectrum between two pixels for BGR channels.
def calculateDistance(img, seed_x, seed_y, child_x, child_y):
    seed_pixel = img[seed_y][seed_x]
    child_pixel = img[child_y][child_x]

    return abs(int(seed_pixel) - int(child_pixel))

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
def blobDetection(img, min_pixels_n, height, width):
    stack = []
    list_of_blobs = []
    pixels = createsAuxMatrix(width, height)
    label = 0
    pixels_per_blob = 0
    blob_px_qty = []

    for y in range(height):
        for x in range(width):

            pix = img[y,x]
            # Does this pixel was never visited before AND is this a foreground? If yes, then we found a seed pixel of a blob
            if pixels[y][x].label == -1 and pix == FOREGROUND:
                pixels[y][x].label = label

                stack.append(pixels[y][x]) # Seed pixel in put inside a stack
                b = Blob()

                # While stack is not empty, verify neighbors from pixel which was never visited before
                while len(stack) != 0:
                    p = stack.pop()

                    # Add it to blob's pixel list
                    b.pixels_list.append(p)
                    pixels_per_blob += 1

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
                else:
                    label -= 1 #returns label

                # Reset values for next iteration
                pixels_per_blob = 0
                label += 1

    return blob_px_qty, list_of_blobs

def update_blob_list(blob_qty_px, blob_list):
    mean = np.mean(blob_qty_px)
    stdev = np.std(blob_qty_px)

    for blob in blob_list:
        std_perc = math.sqrt(pow(blob.pixels_qty - mean, 2))/stdev
        if std_perc >= 2:
            print("std: ", stdev, " mean: ", mean)
            print(blob, 'precisa ser dividido.', blob.pixels_qty)

    return blob_list
            
        
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
    blob_px_qty, list_of_blobs = blobDetection (img_final, MIN_PIXELS_N, img_final.shape[0], img_final.shape[1])
    print ('%d componentes detectados.' % len(list_of_blobs))
    
    # Detects if a blob is 
    list_of_blobs = update_blob_list(blob_px_qty, list_of_blobs)

    cv2.imshow('02 - out', img_final)
    cv2.imwrite('02 - out.png', img_final*255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main ()

#===============================================================================
