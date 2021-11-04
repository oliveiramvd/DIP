import sys
import cv2
import numpy as np

INPUT_IMAGE =  'img/GT2.bmp'
CONTRAST = 0.25
BRIGHTNESS = 0
LOWER_LUMINANCE = 128
UPPER_LUMINANCE = 255

def main ():

    img = cv2.imread (INPUT_IMAGE)
    if img is None:
        print ('Error while opening image.\n')
        sys.exit ()

    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower_t = np.array([0,LOWER_LUMINANCE,0])
    upper_t = np.array([255,UPPER_LUMINANCE,255])
    mask = cv2.inRange(img_hls, lower_t, upper_t)
    res = cv2.bitwise_and(img,img, mask= mask)

    # Mask - Gaussian blur
    g_blurred1 = cv2.GaussianBlur(res, (51, 51), 0)
    g_blurred2 = cv2.GaussianBlur(res, (103, 103), 0)
    g_blurred3 = cv2.GaussianBlur(res, (207, 207), 0)
    g_blurred4 = cv2.GaussianBlur(res, (515, 515), 0)

    # Sum gaussian masks
    g_mask_sum = cv2.add(g_blurred1, g_blurred2)
    g_mask_sum = cv2.add(g_mask_sum, g_blurred3)
    g_mask_sum = cv2.add(g_mask_sum, g_blurred4)

    # Mask - Mean blur
    m_blurred1 = cv2.blur(res, (21,21))
    m_blurred1 = cv2.blur(m_blurred1, (21,21))
    m_blurred2 = cv2.blur(res, (43,43))
    m_blurred2 = cv2.blur(m_blurred2, (43,43))
    m_blurred3 = cv2.blur(res, (83,83))
    m_blurred3 = cv2.blur(m_blurred3, (83,83))
    m_blurred4 = cv2.blur(res, (195,195))
    m_blurred4 = cv2.blur(m_blurred4, (195,195))

    # Sum mean masks
    m_mask_sum = cv2.add(m_blurred1, m_blurred2)
    m_mask_sum = cv2.add(m_mask_sum, m_blurred3)
    m_mask_sum = cv2.add(m_mask_sum, m_blurred4)
   

    # Apply contrast and brightness to the mask
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            m_mask_sum[y, x] = m_mask_sum [y, x] * CONTRAST + BRIGHTNESS
            g_mask_sum[y, x] = g_mask_sum [y, x] * CONTRAST + BRIGHTNESS

    final_mean = cv2.add(img, m_mask_sum)
    final_gaussian = cv2.add(img, g_mask_sum)

    cv2.imshow ('1 - Original', img)
    cv2.imshow ('2 - Bloom (Gaussian Blur)', final_gaussian)
    cv2.imshow ('3 - Bloom (Median blur)', final_mean)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()
