import numpy as np
from numpy import linalg as la

def qr(a):
    n = len(a)
    q = np.empty(shape=[n,n])
    r = np.zeros(shape=[n,n])
    for i in range(n):
        aPerpendicular = a[:,i]
        if(i>0):
            for j in range(i):
                y = q[:,j]
                r[j][i] = np.dot(a[:,i],y)
                z = r[j][i]*y
                aPerpendicular = aPerpendicular - z
        r[i][i] = np.sqrt(np.dot(aPerpendicular,aPerpendicular))
        q[:,i] = aPerpendicular/r[i][i]
    return q,r

def givenRot(a):
    n = len(a)
    for j in range(n-2):
        for i in range(j+2,n):
            gRot = np.identity(n)
            alph = np.sqrt(np.power(a[i][j],2)+np.power(a[j+1][j],2))
            c = a[j+1][j]/alph
            s = a[i][j]/alph
            gRot[i][i] = c
            gRot[j+1][j+1] = c
            gRot[i][j+1] = -s
            gRot[j+1][i] = s
            a = np.matmul(np.matmul(gRot,a),gRot.transpose())
    return a  

  


# a = np.random.rand(8,8)
# # a = np.random.randint(99,size=(100,100))
# a = np.random.randint(100,size=(100,100))

def eig(a,maxiter=1000):
    n = len(a)
    #aTemp = givenRot(a)
    aTemp = a
    eigenvec = np.identity(n)
    for i in range(maxiter):
        q, r = qr(aTemp)
        aTemp = np.matmul(r,q)
        eigenvec = np.matmul(eigenvec,q)
        
    eigenval = np.diag(aTemp)

    x = []

    for i in range(n):
        x.append((eigenval[i],eigenvec[i]))

    return sorted(x,key=lambda p: abs(p[0]))[::-1]

# b,c = qr(a)
# print(np.matmul(b,c))

# for i in range(5):
#     print(a[i][i])

# print(po)

# b = np.random.rand(5,5)

# # gRot = np.zeros(shape=[3,3])
# # for i in range(3):
# #     gRot[i][i] = 1

# # alph = np.sqrt(b[1][0]**2+b[0][0]**2)
# # c = b[0][0]*alph/(b[1][0]**2+b[0][0]**2)
# # s = np.sqrt(1-c**2)
# # print(c,s)
# # gRot[1][1] = c
# # gRot[0][0] = c
# # gRot[1][0] = -s
# # gRot[0][1] = s
# # print(gRot)

def sorting(numbers_array):
    return sorted(numbers_array, key = abs)

a = np.array([[80,4,22,1,40,63,8],[24,41,6,53,2,41,2],
                [9,24,6,-38,3,27,19],[-23,-29,-54,29,11,67,47],
                [1,82,10,12,8,23,28],[-12,43,3,-22,72,21,36],
                [11,12,1,35,1,86,90]])

# # print(np.matmul(gRot,b))
# # print(eig(a,100))
# # print(eig(a,500))
# # print(eig(a,700))
# # print(eig(a,10000))

# f = eig(a)

# s,t = la.eig(a)
# print((s[0],t[0]))
# # print(t)
# print(f[0])
# print(f[0])

# # print(givenRot(b))