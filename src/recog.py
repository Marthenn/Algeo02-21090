import cv2
import numpy as np
from otf import *
from eigen import *
from eigenface import *
from ymldb import *
from util import *

# mencari eigenface dari test image yang dinormalisasi dengan mean_face
def get_test_coeff(path,mean_face):
    test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (80, 80), interpolation = cv2.INTER_AREA) / 255
    test_image = test_image.flatten()
    p = np.empty(shape=[1,len(test_image)])
    p[0,:] = test_image
    test_image = p
    test_normalized = get_mean_diff_array(test_image,mean_face)
    test_eigenface = eigenfaces(test_normalized)
    test_eigenface = test_eigenface.T
    test_eigenface = test_eigenface.flatten()
    test_eigenface = normalize_image_value(test_eigenface)
    print(test_eigenface)
    return test_eigenface
    # test_weight = np.multiply(test_eigenface,test_normalized)
    # print(test_weight)
    # print(test_weight.shape)
    # return test_weight

# def get_test_coeff(path,mean_face):
#     test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     test_image = cv2.resize(test_image, (80, 80), interpolation = cv2.INTER_AREA) / 255
#     test_image = test_image.flatten()
#     p = np.empty(shape=[1,len(test_image)])
#     p[0,:] = test_image
#     test_image = p
#     test_normalized = get_mean_diff_array(test_image,mean_face)
#     test_eigenface = eigenfaces(test_normalized)
#     test_eigenface = test_eigenface
#     test_weight = np.array([np.dot(test_eigenface[i],test_normalized) for i in range(len(test_eigenface))])
#     print(test_weight)
#     print(test_weight.shape)
#     return test_weight

# mencari jarak euclidean antara dua array
def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))

# mencari jarak euclidean antara test image dengan eigenface hasil training minimum
def find_min_euclid(test,data,treshold=0.2):
    data = data['eigen-face']
    min = euclidean(test,normalize_image_value(data[0])) #asumsi terdapat setidaknya satu data
    minIdx = 0 #index lokasi minimum sekarang
    for i in range(1,len(data)):
        temp = euclidean(test,normalize_image_value(data[i]))
        if temp < min:
            min = temp
            minIdx = i
        #print(min)
    min /= 255**2
    print(min)
    if min<treshold:
        return minIdx
    else:
        return None #kalau min di atas treshold maka gk dapat apa"

# cari pasangan yang cocok
def find_match(test_path,data):
    test_weight = get_test_coeff(test_path,data['mean-face'])
    idx = find_min_euclid(test_weight,data)
    if(idx != None):
        print(idx)
    else:
        print("tidak ditemukan")

# cari treshold
def get_treshold(data):
    data = data['recognized-face'] # ini buat ntar access weightnya
    max = euclidean(data[0],data[1])
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            temp = euclidean(data[i],data[j])
            if temp > max:
                max = temp
    return 0.5*max

# test code using some samples image
if __name__ == '__main__':
    data = read_from_yml("D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\doc","db.yml")
    find_match(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\test_komukGambarAI.png",data)