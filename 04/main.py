import math
import numpy as np
import cv2

from matplotlib import pyplot as plt
from numpy.core.fromnumeric import std
from classes import Pixel
from classes import Blob

INPUT_IMAGE = ["img/60.bmp", "img/82.bmp", "img/114.bmp", "img/150.bmp","img/205.bmp"]
MIN_PIXELS_N = 15               # Minimum blob pixel quantity
FOREGROUND = 255                # Color to fill in Foreground (0 - 255)
BACKGROUND = 0                  # Color to fill in Background (0 - 255)

# Calculates the distance in color spectrum between two pixels for BGR channels.
def calculates_distance(img, seed_x, seed_y, child_x, child_y):
    seed_pixel = img[seed_y][seed_x]
    child_pixel = img[child_y][child_x]

    return abs(int(seed_pixel) - int(child_pixel))

# Finds the minimum and maximum positions of the edge pixels of the region of interest in the image
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
def creates_aux_matrix(width, height):
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
def blob_detection(img, min_pixels_n):
    height, width = img.shape[0], img.shape[1]
    stack = []
    list_of_blobs = []
    pixels = creates_aux_matrix(width, height)
    label = 0
    pixels_per_blob = 0
    blob_px_qty = []

    # Store edge pixels within the image
    global_min_max_pos = {'min': {'x': width-1, 'y': height-1 }, 
                        'max': {'x': 0, 'y': 0}}

    for y in range(height):
        for x in range(width):

            pix = img[y,x]
            # Does this pixel was never visited before AND is this a foreground? If yes, then we found a seed pixel of a blob
            if pixels[y][x].label == -1 and pix == FOREGROUND:
                pixels[y][x].label = label

                stack.append(pixels[y][x])
                b = Blob(width, height)

                # While stack is not empty, verify neighborhood from pixel which was never visited before
                while len(stack) != 0:
                    p = stack.pop()

                    # Add it to blob's pixel list
                    b.pixels_list.append(p)
                    pixels_per_blob += 1
                    # checks if this new pixel is a new min/max position in that blob 
                    b = find_min_max_pixel_position(b, p)

                    # Neighborhood 4
                    # Top
                    if p.y - 1 >= 0:
                        if pixels[p.y-1][p.x].label == -1:
                            dist = calculates_distance(img, p.x, p.y, p.x, p.y-1)
                            if dist == 0:
                                pixels[p.y-1][p.x].label = label
                                stack.append(pixels[p.y-1][p.x])
                    # Right
                    if p.x + 1 < width:
                        if pixels[p.y][p.x+1].label == -1:
                            dist = calculates_distance(img, p.x, p.y, p.x+1, p.y)
                            if dist == 0:
                                pixels[p.y][p.x+1].label = label
                                stack.append(pixels[p.y][p.x+1])
                    # Bottom
                    if p.y + 1 < height:
                        if pixels[p.y+1][p.x].label == -1:
                            dist = calculates_distance(img, p.x, p.y, p.x, p.y+1)
                            if dist == 0:
                                pixels[p.y+1][p.x].label = label
                                stack.append(pixels[p.y+1][p.x])
                    # Left
                    if p.x - 1 >= 0:
                        if pixels[p.y][p.x-1].label == -1:
                            dist = calculates_distance(img, p.x, p.y, p.x-1, p.y)
                            if dist == 0:
                                pixels[p.y][p.x-1].label = label
                                stack.append(pixels[p.y][p.x-1])

                # If there is no more pixels in the stack, then all pixels from this blob were found
                b.pixels_qty = pixels_per_blob

                # If blob doesn't have minimum qty of pixels, ignore it
                if b.pixels_qty >= min_pixels_n:
                    list_of_blobs.append(b)
                    blob_px_qty.append(pixels_per_blob)

                    # Checks if it's a global min/max edge pixel
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

# Crop region of interest and resize it within the image (it will be used to adjust all images to have the same ratio)     
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

    img_resize = cv2.resize(img_crop, (int(img_crop.shape[1] + original_width * new_proportion), int(img_crop.shape[0] + original_height * new_proportion)), interpolation=cv2.INTER_LINEAR)

    return img_resize

# Adjusts the amount of blobs in each image according to a measure of dispersion
def adjusts_blobs_from_list(list_blobs_per_image, global_px_per_blob):
    std_dev = np.std(global_px_per_blob)
    mean = np.mean(global_px_per_blob)

    for i in range(len(list_blobs_per_image)):
        # Checks if a blob is greater than a standard deviation
        for blob in list_blobs_per_image[i]:
            if blob > mean + std_dev:

                # Breaks a blob until it's within one standard deviation from the mean
                while blob > mean + std_dev:
                    blob -= mean
                    list_blobs_per_image[i].append(mean)

    return list_blobs_per_image

# Filters and binarizes a given image
def binarization(img):
    img_copy = img.copy()

    # Apply image normalization
    aux_norm = np.zeros((800,800))
    img_norm = cv2.normalize(img_copy, aux_norm, 0, 255, cv2.NORM_MINMAX)

    # Apply median blur
    img_blur = cv2.medianBlur(img_norm,5)

    # Adaptive thresholding
    img_bin = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51,-30)

    return img_bin

def main ():

    img_final = []
    global_px_per_blob = []
    list_blobs_per_image = []

    # Reads all images in the list
    for i in range(len(INPUT_IMAGE)):
        img = cv2.imread(INPUT_IMAGE[i], cv2.IMREAD_GRAYSCALE)
        img_bin = binarization(img)

        # Segmentation for image resizing (discarted, since it doesn't make difference in the final result anyway~~)
        #global_min_max_pos, blob_px_qty, list_of_blobs = blob_detection(img_bin, MIN_PIXELS_N)
        #img_res = image_resize(img_bin.copy(), global_min_max_pos)

        # Image segmentation and gets the global quantity of all blobs from all the images
        global_min_max_pos, blob_px_qty, list_of_blobs = blob_detection(img_bin, MIN_PIXELS_N)

        global_px_per_blob+= blob_px_qty
        list_blobs_per_image.append(blob_px_qty)

        img_final.append(img_bin)

    list_blobs_per_image = adjusts_blobs_from_list(list_blobs_per_image, global_px_per_blob)

    # Lists all blobs from all the images
    for i in range(len(list_blobs_per_image)):
        print("Image '"+str(INPUT_IMAGE[i]) + "': " +  str(len(list_blobs_per_image[i])))


if __name__ == '__main__':
    main ()
