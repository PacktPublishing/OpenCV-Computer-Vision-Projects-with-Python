import os
import sys
import argparse
import cPickle as pickle
import json

import cv2
import numpy as np
from sklearn.cluster import KMeans

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Creates features for given images')
    parser.add_argument("--samples", dest="cls", nargs="+", action="append", 
            required=True, help="Folders containing the training images. \
            The first element needs to be the class label.")
    parser.add_argument("--codebook-file", dest='codebook_file', required=True,
            help="Base file name to store the codebook")
    parser.add_argument("--feature-map-file", dest='feature_map_file', required=True,
            help="Base file name to store the feature map")
    parser.add_argument("--scale-image", dest="scale", type=int, default=150,
            help="Scales the longer dimension of the image down to this size.")

    return parser

def load_input_map(label, input_folder):
    combined_data = []

    if not os.path.isdir(input_folder):
        print "The folder " + input_folder + " doesn't exist"
        raise IOError
        
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})
                    
    return combined_data

class FeatureExtractor(object):
    def extract_image_features(self, img):
        kps = DenseDetector().detect(img)
        kps, fvs = SIFTExtractor().compute(img, kps)
        return fvs

    def get_centroids(self, input_map, num_samples_to_fit=10):
        kps_all = []
        
        count = 0
        cur_label = ''
        for item in input_map:
            if count >= num_samples_to_fit:
                if cur_label != item['label']:
                    count = 0
                else:
                    continue

            count += 1

            if count == num_samples_to_fit:
                print "Built centroids for", item['label']

            cur_label = item['label']
            img = cv2.imread(item['image'])
            img = resize_to_size(img, 150)

            num_dims = 128
            fvs = self.extract_image_features(img)
            kps_all.extend(fvs) 

        kmeans, centroids = Quantizer().quantize(kps_all)
        return kmeans, centroids

    def get_feature_vector(self, img, kmeans, centroids):
        return Quantizer().get_feature_vector(img, kmeans, centroids)

def extract_feature_map(input_map, kmeans, centroids):
    feature_map = []
     
    for item in input_map:
        temp_dict = {}
        temp_dict['label'] = item['label']
    
        print "Extracting features for", item['image']
        img = cv2.imread(item['image'])
        img = resize_to_size(img, 150)

        temp_dict['feature_vector'] = FeatureExtractor().get_feature_vector(
                    img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)

    return feature_map

class Quantizer(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.extractor = SIFTExtractor()
        self.num_clusters = num_clusters
        self.num_retries = 10

    def quantize(self, datapoints):
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)

        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def get_feature_vector(self, img, kmeans, centroids):
        kps = DenseDetector().detect(img)
        kps, fvs = self.extractor.compute(img, kps)
        labels = kmeans.predict(fvs)
        fv = np.zeros(self.num_clusters)

        for i, item in enumerate(fvs):
            fv[labels[i]] += 1

        fv_image = np.reshape(fv, ((1, fv.shape[0])))
        return self.normalize(fv_image)

class DenseDetector(object):
    def __init__(self, step_size=20, feature_scale=40, img_bound=20):
        self.detector = cv2.FeatureDetector_create("Dense")
        self.detector.setInt("initXyStep", step_size)
        self.detector.setInt("initFeatureScale", feature_scale)
        self.detector.setInt("initImgBound", img_bound)

    def detect(self, img):
        return self.detector.detect(img)

class SIFTExtractor(object):
    def compute(self, image, kps):
        if image is None:
            print "Not a valid image"
            raise TypeError

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = cv2.SIFT().compute(gray_image, kps)
        return kps, des

# Resize the shorter dimension to 'new_size'
# while maintaining the aspect ratio
def resize_to_size(input_image, new_size=150):
    h, w = input_image.shape[0], input_image.shape[1]
    ds_factor = new_size / float(h)
    if w < h:
        ds_factor = new_size / float(w)
    new_size = (int(w * ds_factor), int(h * ds_factor))
    return cv2.resize(input_image, new_size) 

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    
    input_map = []
    for cls in args.cls:
        assert len(cls) >= 2, "Format for classes is `<label> file`"
        label = cls[0]
        input_map += load_input_map(label, cls[1])

    downsample_length = args.scale

    # Building the codebook
    print "===== Building codebook ====="
    kmeans, centroids = FeatureExtractor().get_centroids(input_map)
    if args.codebook_file:
        with open(args.codebook_file, 'w') as f:
            pickle.dump((kmeans, centroids), f)
    
    # Input data and labels
    print "===== Building feature map ====="
    feature_map = extract_feature_map(input_map, kmeans, centroids)
    if args.feature_map_file:
        with open(args.feature_map_file, 'w') as f:
            pickle.dump(feature_map, f)

