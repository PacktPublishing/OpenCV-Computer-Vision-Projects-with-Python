from sfm import StructureFromMotion
import numpy as np


def main():
    # camera matrix and distortion coefficients
    # can be recovered with calibrate.py
    # but the examples used here are already undistorted, taken with a camera
    # of known K
    K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4,
                   1006.81/4, 0, 0, 1]]).reshape(3, 3)
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    SfM = StructureFromMotion(K, d)

    # load a pair of images for which to perform SfM
    SfM.LoadImagePair("fountain_dense/0004.png", "fountain_dense/0005.png")

    # SfM.PlotOpticFlow()

    # SfM.DrawEpipolarLines()

    # SfM.PlotRectifiedImages()

    SfM.PlotPointCloud()


if __name__ == '__main__':
    main()
