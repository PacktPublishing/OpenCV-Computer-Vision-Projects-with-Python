import cv2
import numpy as np

img = cv2.imread('../images/input_harris_GFTT_box.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

# for the polygon image
#dst = cv2.cornerHarris(gray,4,5,0.02)

# for the box image 
dst = cv2.cornerHarris(gray,4,5,0.04)       # limited corners
#dst = cv2.cornerHarris(gray,9,5,0.04)      # bottom right corner
#dst = cv2.cornerHarris(gray,14,5,0.04)     # top left corner

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,0]

cv2.imshow('Harris Corners',img)
cv2.waitKey()
