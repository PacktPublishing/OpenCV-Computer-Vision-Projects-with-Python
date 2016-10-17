import cv2
import numpy as np

input_image = cv2.imread('../images/input_sift_surf_orb_fishing.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) 

# Initiate ORB object
orb = cv2.ORB()

# find the keypoints with ORB
keypoints = orb.detect(gray_image, None)

# compute the descriptors with ORB
keypoints, descriptors = orb.compute(gray_image, keypoints)

# draw only the location of the keypoints without size or orientation
final_keypoints = cv2.drawKeypoints(input_image, keypoints, color=(0,255,0), flags=0)

cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()
