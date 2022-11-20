import sys
import os
sys.path.insert(0,os.path.abspath(os.curdir))

from src.trainer.otf import *
from src.trainer.eigen import eig
from numpy import linalg as la

def getEigenface(A,vx):
    n = len(vx)
    vi = np.empty(shape=[n,1])
    vi[:,0] = vx #/la.norm(vx)
    face = np.matmul(A,vi)[:,0] #/ la.norm(np.matmul(A,vi)[:,0])

    return face

def eigenfaces(A): #terima matrix hasil get_mean_diff_array 
    coeff_arr = get_compressed_cov_array(A)
    
    a = eig(coeff_arr,50)
    V = (list(zip(*a))[1])
    
    A = A.T

    n = len(A)
    m = len(A[0])
    faces = np.empty(shape=[n,m])
    for i in range(m):
        faces[:,i] = getEigenface(A,V[i])
    
    return faces
#return adalah matrix berisi eigenface tiap kolom

#contoh penggunaan
# x array hasil get mean diff array 
# a = eigenfaces(x)
# a = a.T

# for i in range(105):
#     cv2.imwrite('/home/archkoi/Desktop/alg/Algeo02-21090/asu/abc{}.png'.format(i),(((a[i]+g)*255)).reshape(100,100))
