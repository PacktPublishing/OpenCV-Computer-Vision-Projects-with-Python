import cv2
import numpy as np

img = cv2.imread('../images/input_train.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

sobel_horizontal_1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_horizontal_2 = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=5)
sobel_horizontal_3 = cv2.Sobel(img, cv2.CV_64F, 3, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
canny = cv2.Canny(img, 50, 240)

cv2.imshow('Original', img)
#cv2.imshow('Sobel horizontal 1', sobel_horizontal_1)
#cv2.imshow('Sobel horizontal 2', sobel_horizontal_2)
#cv2.imshow('Sobel horizontal 3', sobel_horizontal_3)
#cv2.imshow('Sobel vertical', sobel_vertical)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Canny', canny)

cv2.waitKey()
