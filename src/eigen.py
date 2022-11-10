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
a = np.arange(10000).reshape(100,100)

def eig(a,maxiter=1000):
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

print(eig(a))