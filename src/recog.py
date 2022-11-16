import os, cv2
import numpy as np
from otf import *
from eigen import *
from eigenface import *

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
    test_weight = np.multiply(test_eigenface,test_normalized)
    return test_weight

# mencari jarak euclidean antara dua array
def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))

# mencari jarak euclidean antara test image dengan eigenface hasil training minimum
def find_min_euclid(test,data,treshold=0.2):
    min = euclidean(test,data[0]) #asumsi terdapat setidaknya satu data
    minIdx = 0 #index lokasi minimum sekarang
    for i in range(1,len(data)):
        temp = euclidean(test,data[i])
        if temp < min:
            min = temp
            minIdx = i
    if min<treshold:
        return minIdx
    else:
        return None #kalau min di atas treshold maka gk dapat apa"

# cari 
def find_match(test_path,data_path):
    print('masuk ke find math')
    test_weight = get_test_coeff(test_path,0)
    print('udah dapat test weight')
    idx = find_min_euclid(test_weight,data_path)
    print('dapat indeks')
    if(idx != None):
        print(idx)
    else:
        print("tidak ditemukan")

# if __name__ == '__main__':
#     data = np.array([])
#     data_0 = cv2.imread(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\aaa\eigenface-0.jpg", cv2.IMREAD_GRAYSCALE)
#     data_0 = cv2.resize(data_0, (16, 16), interpolation = cv2.INTER_AREA) / 255
#     data_0 = data_0.flatten()
#     data = np.append(data,data_0)
#     data_1 = cv2.imread(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\aaa\eigenface-1.jpg", cv2.IMREAD_GRAYSCALE)
#     data_1 = cv2.resize(data_1, (16, 16), interpolation = cv2.INTER_AREA) / 255
#     data_1 = data_1.flatten()
#     data = np.append(data,data_1)
#     data_2 = cv2.imread(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\aaa\eigenface-2.jpg", cv2.IMREAD_GRAYSCALE)
#     data_2 = cv2.resize(data_2, (16, 16), interpolation = cv2.INTER_AREA) / 255
#     data_2 = data_2.flatten()
#     data = np.append(data,data_2)
#     data_3 = cv2.imread(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\aaa\eigenface-3.jpg", cv2.IMREAD_GRAYSCALE)
#     data_3 = cv2.resize(data_3, (16, 16), interpolation = cv2.INTER_AREA) / 255
#     data_3 = data_3.flatten()
#     data = np.append(data,data_3)
#     data_4 = cv2.imread(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\aaa\eigenface-4.jpg", cv2.IMREAD_GRAYSCALE)
#     data_4 = cv2.resize(data_4, (16, 16), interpolation = cv2.INTER_AREA) / 255
#     data_4 = data_4.flatten()
#     data = np.append(data,data_4)
#     data_5 = cv2.imread(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\aaa\eigenface-5.jpg", cv2.IMREAD_GRAYSCALE)
#     data_5 = cv2.resize(data_5, (16, 16), interpolation = cv2.INTER_AREA) / 255
#     data_5 = data_5.flatten()
#     data = np.append(data,data_5)
#     print("data sudah beres")
#     print(data)
#     print(find_match(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\test.jpg",data))