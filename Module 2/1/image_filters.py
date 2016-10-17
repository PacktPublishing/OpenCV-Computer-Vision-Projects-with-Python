# http://lodev.org/cgtutor/filtering.html

import cv2
import numpy as np

#img = cv2.imread('../images/input_sharp_edges.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('../images/input_tree.jpg')
rows, cols = img.shape[:2]
#cv2.imshow('Original', img)

###################
# Motion Blur
size = 15
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
output = cv2.filter2D(img, -1, kernel_motion_blur)
#cv2.imshow('Motion Blur', output)

###################
# Sharpening
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1], 
                             [-1,2,2,2,-1], 
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
#cv2.imshow('Sharpening', output_1)
#cv2.imshow('Excessive Sharpening', output_2)
#cv2.imshow('Edge Enhancement', output_3)

###################
# Embossing
img_emboss_input = cv2.imread('../images/input_house.jpg')
kernel_emboss_1 = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],
                            [-1,0,1],
                            [0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],
                            [0,0,0],
                            [0,0,-1]])
gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY)
output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1)
output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2)
output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3)
cv2.imshow('Input', img_emboss_input) 
cv2.imshow('Embossing - South West', output_1 + 128)
cv2.imshow('Embossing - South East', output_2 + 128)
cv2.imshow('Embossing - North West', output_3 + 128)

###################
# Erosion and dilation

img = cv2.imread('../images/input_morphology.png',0)
kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
#cv2.imshow('Input', img)
#cv2.imshow('Erosion', img_erosion)
#cv2.imshow('Dilation', img_dilation)

cv2.waitKey()

