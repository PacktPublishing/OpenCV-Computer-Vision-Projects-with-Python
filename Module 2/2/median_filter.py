import cv2
import numpy as np

img = cv2.imread('../images/input_median_filter.png')
output = cv2.medianBlur(img, 7)
cv2.imshow('Input', img)
cv2.imshow('Median filter', output)
cv2.waitKey()
