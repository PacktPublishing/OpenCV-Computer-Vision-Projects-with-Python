import cv2
import numpy as np

img = cv2.imread('../images/input_bilateral_filter.jpg')
img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)

img_gaussian = cv2.GaussianBlur(img, (13,13), 0)
img_bilateral = cv2.bilateralFilter(img, 13, 70, 50)

cv2.imshow('Input', img)
cv2.imshow('Gaussian filter', img_gaussian)
cv2.imshow('Bilateral filter', img_bilateral)
cv2.waitKey()
