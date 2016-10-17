import sys

import cv2
import numpy as np

def get_all_contours(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    return contours

if __name__=='__main__':
    #img = cv2.imread('../images/input_nike_logo_shapes.png')
    img = cv2.imread(sys.argv[1])
    input_contours = get_all_contours(img)

    factor = 0.00001
    while True:
        output_img = np.zeros(img.shape, np.uint8) + 255

        for contour in input_contours: 
            epsilon = factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.drawContours(output_img, [approx], -1, (0,0,0), 3)

        cv2.imshow('Output', output_img)
        c = cv2.waitKey()
        if c == 27:
            break
             
        factor *= 0.75
                
