import os, cv2
import numpy as np
from otf import *
from eigen import *

path = r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\test.jpg"

def get_test_coeff(path):
    test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (40, 40), interpolation = cv2.INTER_AREA) / 255
    test_image = get_coeff_array(test_image)
    test_image = eig(test_image)
    test_image.sort()
    eigVal = np.array([])
    for eVal in test_image:
        eigVal = np.append(eigVal, eVal[0])
    return eigVal

def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))