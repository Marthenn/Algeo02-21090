import os, cv2
import numpy as np
from otf import *
from eigen import *

path = r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\test.jpg"

# mencari eigenface dari test image yang dinormalisasi dengan mean_face
def get_test_coeff(path,mean_face):
    test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (40, 40), interpolation = cv2.INTER_AREA) / 255
    test_image = get_coeff_array(test_image) - mean_face
    test_image = eig(test_image)
    eigVal = np.array([])
    for eVal in test_image:
        eigVal = np.append(eigVal, eVal[0])
    return eigVal

# mencari jarak euclidean antara dua array
def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))

# mencari jarak euclidean antara test image dengan eigenface hasil training minimum
def find_min_euclid(test,data):
    min = euclidean(test,data[0])
    minIdx = 0
    for i in range(1,len(data)):
        temp = euclidean(test,data[i])
        if temp < min:
            min = temp
            minIdx = i
    return minIdx