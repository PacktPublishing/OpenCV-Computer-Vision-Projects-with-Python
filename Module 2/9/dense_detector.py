import cv2
import numpy as np

class DenseDetector(object):
    def __init__(self, step_size=20, feature_scale=40, img_bound=20):
        self.detector = cv2.FeatureDetector_create("Dense")
        self.detector.setInt("initXyStep", step_size)
        self.detector.setInt("initFeatureScale", feature_scale)
        self.detector.setInt("initImgBound", img_bound)

    def detect(self, img):
        return self.detector.detect(img)

if __name__=='__main__':
    input_image = cv2.imread('../images/input_dense_detector.jpg')
    input_image_sift = np.copy(input_image)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    #keypoints = DenseDetector(50,15,5).detect(input_image)
    keypoints = DenseDetector(20,20,5).detect(input_image)
    input_image = cv2.drawKeypoints(input_image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Dense feature detector', input_image)

    sift = cv2.SIFT()
    keypoints = sift.detect(gray_image, None)
    input_image_sift = cv2.drawKeypoints(input_image_sift, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('SIFT detector', input_image_sift)

    cv2.waitKey()

