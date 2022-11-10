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
            gRot = np.zeros(shape=[n,n])
            for k in range(n):
                gRot[k][k] = 1
            alph = np.sqrt(a[i][j]**2+a[j+1][j]**2)
            c = a[j+1][j]/alph
            s = a[i][j]/alph
            gRot[i][i] = c
            gRot[j+1][j+1] = c
            gRot[i][j+1] = -s
            gRot[j+1][i] = s
            a = np.matmul(np.matmul(gRot,a),gRot.transpose())
    return a    

# def QR_Decomposition(A):
#     n, m = A.shape # get the shape of A
#     Q = np.empty((n, n)) # initialize matrix Q
#     u = np.empty((n, n)) # initialize matrix u
#     u[:, 0] = A[:, 0]
#     Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

#     for i in range(1, n):

#         u[:, i] = A[:, i]
#         for j in range(i):
#             u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector
#         Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor
#     R = np.zeros((n, m))
#     for i in range(n):
#         for j in range(i, m):
#             R[i, j] = A[:, j] @ Q[:, i]

#     return Q, R

# def QR_eigvals(A, tol=1e-12, maxiter=1000):
#     A_old = np.copy(A)
#     A_new = np.copy(A)

#     diff = np.inf
#     i = 0
#     while (diff > tol) and (i < maxiter):
#         A_old[:, :] = A_new
#         Q, R = QR_Decomposition(A_old)

#         A_new[:, :] = R @ Q

#         diff = np.abs(A_new - A_old).max()
#         i += 1
#     eigvals = np.diag(A_new)

#     return eigvals


# a = np.random.rand(8,8)
# a = np.random.randint(99,size=(100,100))
a = np.random.randint(100,size=(100,100))

def eig(a,maxiter=1000):
    a = givenRot(a)
    for i in range(maxiter):
        b, c = qr(a)
        a = np.matmul(c,b)
    eigenval = np.empty(len(a))
    for i in range(len(a)):
        eigenval[i] = a[i][i]
    
    return eigenval

# b,c = qr(a)
# print(np.matmul(b,c))

# for i in range(5):
#     print(a[i][i])

# print(po)

b = np.random.rand(5,5)

# gRot = np.zeros(shape=[3,3])
# for i in range(3):
#     gRot[i][i] = 1

# alph = np.sqrt(b[1][0]**2+b[0][0]**2)
# c = b[0][0]*alph/(b[1][0]**2+b[0][0]**2)
# s = np.sqrt(1-c**2)
# print(c,s)
# gRot[1][1] = c
# gRot[0][0] = c
# gRot[1][0] = -s
# gRot[0][1] = s
# print(gRot)

a = np.array([[80,4,22,1,40,63,8],[24,41,6,53,2,41,2],
                [9,24,6,-38,3,27,19],[-23,-29,-54,29,11,67,47],
                [1,82,10,12,8,23,28],[-12,43,3,-22,72,21,36],
                [11,12,1,35,1,86,90]])

# print(np.matmul(gRot,b))
print(eig(a,100))
print(eig(a,500))
print(eig(a,700))
print(eig(a,10000))
s,t = la.eig(a)
print((np.real(s)))

# print(givenRot(b))