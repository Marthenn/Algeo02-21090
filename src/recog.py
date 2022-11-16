import os, cv2
import numpy as np
from otf import *
from eigen import *

# mencari eigenface dari test image yang dinormalisasi dengan mean_face
def get_test_coeff(path,mean_face):
    test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (16, 16), interpolation = cv2.INTER_AREA) / 255
    test_image = test_image.flatten()
    p = np.empty(shape=[1,len(test_image)])
    p[0,:] = test_image
    test_image = p
    test_normalized = get_mean_diff_array(test_image,mean_face)
    coeff_arr = get_coeff_array(test_normalized)
    test_eigen = eig(coeff_arr)
    eigVal = list(list(zip(*test_eigen))[0])
    # eigVal = np.array([])
    # for eVal in test_eigen:
    #     eigVal = np.append(eigVal, eVal[0])
    # arr = list(zip(*test_eigen))[1][0]/la.norm(list(zip(*test_eigen))[1][0]) * 255
    # np.savetxt('myeigenvecresult.txt',arr,fmt='%.4e')
    # cv2.imwrite('abc.png',arr.reshape(16,16))
    return eigVal

# mencari jarak euclidean antara dua array
def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))

# mencari jarak euclidean antara test image dengan eigenface hasil training minimum
def find_min_euclid(test,data,treshold):
    min = euclidean(test,data[0])
    minIdx = 0
    for i in range(1,len(data)):
        temp = euclidean(test,data[i])
        if temp < min:
            min = temp
            minIdx = i
    if min<treshold:
        return minIdx
    else:
        return None #kalau min di atas treshold maka gk dapat apa"

# path = r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\test.jpg"
# print(get_test_coeff(path,0))