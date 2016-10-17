import os
import sys
import argparse 
import cPickle as pickle 

import cv2
import numpy as np

import create_features as cf
from training import ClassifierTrainer

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extracts features \
            from each line and classifies the data')
    parser.add_argument("--input-image", dest="input_image", required=True,
            help="Input image to be classified")
    parser.add_argument("--svm-file", dest="svm_file", required=True,
            help="File containing the trained SVM model")
    parser.add_argument("--codebook-file", dest="codebook_file", 
            required=True, help="File containing the codebook")
    return parser

class ImageClassifier(object):
    def __init__(self, svm_file, codebook_file):
        with open(svm_file, 'r') as f:
            self.svm = pickle.load(f)

        with open(codebook_file, 'r') as f:
            self.kmeans, self.centroids = pickle.load(f)

    def getImageTag(self, img):
        img = cf.resize_to_size(img)
        feature_vector = cf.FeatureExtractor().get_feature_vector(img, self.kmeans, self.centroids)
        image_tag = self.svm.classify(feature_vector)
        return image_tag

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    svm_file = args.svm_file
    codebook_file = args.codebook_file
    input_image = cv2.imread(args.input_image)

    print "Output class:", ImageClassifier(svm_file, codebook_file).getImageTag(input_image)
