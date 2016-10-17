import cv2
import numpy as np

img = cv2.imread('../images/input_histogram.jpg', 0)
histeq = cv2.equalizeHist(img)

#cv2.imshow('Input', img)
#cv2.imshow('Histogram equalized', histeq)

##################
# Histogram equalization of color images

img = cv2.imread('../images/input_histogram_color.jpg')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)

cv2.waitKey()

