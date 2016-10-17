import cv2
import numpy as np

input_image = cv2.imread('../images/input_sift_surf_orb_fishing.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
keypoints = sift.detect(gray_image, None)

input_image = cv2.drawKeypoints(input_image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT features', input_image)
cv2.waitKey()

######

# To detect and compute at the same time
#keypoints, descriptors = sift.detectAndCompute(gray_image, None)
