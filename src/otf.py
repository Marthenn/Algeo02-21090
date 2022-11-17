import os

import numpy as np
import cv2
import util

def get_pict_array(url):
    array = []
    first = True
    print("start crawling")
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(url)):
        if dirpath == url:
            continue

        print("crawling:", dirpath)

        for j, file in enumerate(filenames):
            image = cv2.imread(os.path.join(dirpath, file), cv2.IMREAD_GRAYSCALE)
            # normalize pixels values to floating point range [0..1] inclusive
            image = cv2.resize(image, (util.get_image_dim(), util.get_image_dim()), interpolation=cv2.INTER_AREA) / 255

            if first:
                first = False
                array = image.flatten()
            else:
                array = np.vstack((array, image.flatten()))

            break

    return array


def get_mean_vspace(array):
    sh = array.shape

    # if array has length 1, we'll skip calculating the sum since it'll give a one element arr
    try:
        if sh[1] == 1:
            return array

    except IndexError:
        return array

    return np.sum(array, axis=0) / array.shape[0]


def get_mean_diff_array(array, mean_array):
    return array - mean_array


def get_coeff_array(array):
    return np.dot(array.T, array) / array.shape[0]


def get_compressed_cov_array(array):
    return np.dot(array, array.T)


def get_cov_from_file(url):
    pict_arr = get_pict_array(url)

    return get_compressed_cov_array(get_mean_diff_array(pict_arr, get_mean_vspace(pict_arr)))

# pict_arr = get_pict_array('../test/train')
# coeff_arr = get_coeff_array(get_mean_diff_array(pict_arr, get_mean_vspace(pict_arr)))
#
# vects = eigen.eig(coeff_arr)[1]
#
# for i in range(len(vects)):
#     print("printing face")
#     face = np.array(vects[i]).reshape(50, 50)
#     face *= 255
#     print(face)
#     cv2.imwrite("/home/zidane/kuliah/Semester 3/IF2123 - Aljabar Linier dan Geometri/mean_face_.jpg",
#                 face)
#
#     break
