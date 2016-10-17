import cv2
import numpy as np

gray_image = cv2.imread('../images/input_brief.jpg', 0)

# Initialize STAR detector
star = cv2.FeatureDetector_create("STAR")

# Initialize BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")

# find the keypoints with STAR
keypoints = star.detect(gray_image, None)

# compute the descriptors with BRIEF
keypoints, descriptors = brief.compute(gray_image, keypoints)

gray_keypoints = cv2.drawKeypoints(gray_image, keypoints, color=(0,255,0))
cv2.imshow('BRIEF keypoints', gray_keypoints)
cv2.waitKey()
