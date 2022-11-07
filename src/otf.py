import os

import numpy as np
import cv2

def get_pict_array(url):
    array = []
    first = True
    print("start crawling")
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(url)):
        if dirpath == url:
            continue

        print("crawling:", dirpath)

        for i, file in enumerate(filenames):
            image = cv2.imread(os.path.join(dirpath, file), cv2.IMREAD_GRAYSCALE)
            # divide pixel number representative or not?
            image = cv2.resize(image, (120, 80), interpolation = cv2.INTER_AREA) / 255

            if first:
                first = False
                array = image.flatten()
            else:
                array = np.vstack((array, image.flatten()))

            if i == 20:
                break

            break

    return array


def get_mean_vspace(array):
    return np.sum(array, axis=0) / array.shape[0]


def get_mean_diff_array(array, mean_array):
    return array-mean_array


def get_coeff_array(array):
    return np.dot(array.T, array)/array.shape[0]