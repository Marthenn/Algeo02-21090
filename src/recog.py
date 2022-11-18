import cv2
import numpy as np
from otf import *
from eigen import *
from eigenface import *
from ymldb import *
from util import *
from scipy.ndimage import gaussian_filter
import src.imageprocessor.improc as improc

# mencari eigenface dari test image yang dinormalisasi dengan mean_face
def get_test_coeff(path,mean_face,eigenface):
    #test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_image = improc.hist_eq(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    test_image = gaussian_filter(test_image, sigma=3)
    test_image = cv2.resize(test_image, (80, 80), interpolation = cv2.INTER_AREA) / 255
    test_image = test_image.flatten()
    p = np.empty(shape=[1,len(test_image)])
    p[0,:] = test_image
    test_image = p
    test_normalized = get_mean_diff_array(test_image,mean_face).flatten()
    # test_eigenface = eigenfaces(test_normalized)
    # test_eigenface = test_eigenface.T
    # test_eigenface = test_eigenface.flatten()
    # test_eigenface = normalize_image_value(test_eigenface)
    # print(test_eigenface)
    # return test_eigenface
    # test_eigenface = test_eigenface.flatten()
    eigenface = normalize_image_value(eigenface)/255
    test_weight = np.array([np.dot(eigenface[i], test_normalized) for i in range(len(eigenface))])
    # print(test_weight.shape)
    # print("test_weight")
    # print(test_weight)
    # cetak_komuk = np.array([(test_weight[i]*eigenface[i]) for i in range(len(eigenface))])
    # cetak_komuk = np.sum(cetak_komuk,axis=0)
    # print(cetak_komuk.shape)
    # print(mean_face.shape)
    # cetak_komuk = normalize_image_value(cetak_komuk)
    # cv2.imwrite("cetak_komuk.jpg",cetak_komuk.reshape(80,80))
    return test_weight

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
def find_min_euclid(test,data,treshold):
    data = data['recognized-face']
    min = euclidean(test,normalize_image_value(data[0][1])) #asumsi terdapat setidaknya satu data
    tup = data[0] #index lokasi minimum sekarang
    for i in range(1,len(data)):
        temp = euclidean(test,data[i][1])
        if temp < min:
            min = temp
            tup = data[i]
        #print(min)
    print(min)
    if min<treshold:
        return tup
    else:
        return None #kalau min di atas treshold maka gk dapat apa"

# cari pasangan yang cocok
def find_match(test_path,data,treshold):
    test_weight = get_test_coeff(test_path,data['mean-face'],data['eigen-face'])
    return find_min_euclid(test_weight,data,treshold)

# cari treshold
def get_treshold(data):
    data = data['recognized-face'] # ini buat ntar access weightnya
    max = euclidean(data[0][1],data[1][1])
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            temp = euclidean(data[i][1],data[j][1])
            if temp > max:
                max = temp
    return 0.5*max

# test code using some samples image
if __name__ == '__main__':
    data = read_from_yml("D:\Kuliah\Semester 3\AlGeo\Algeo02-21090","db.yml")
    tres = get_treshold(data)
    print(tres)
    res = find_match(r"D:\Kuliah\Semester 3\AlGeo\Algeo02-21090\src\marthen.jpg",data,tres)
    if(res != None):
        print(res[0])
    else:
        print("no match")