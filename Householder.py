import numpy as np

# 2.1) Funció que donada una matriu A i un index i ens retorna
# la matriu de Householder Hi. Adaptem el càlcul de la matriu H_i a python:

def matriuHouseholder(A, i):
    m=len(A)
    I= np.eye(i)
    x=A[i:m, i]
    v=np.copy(x).reshape(len(x), 1)
    v[0]=v[0]+np.sign(x[0])*np.linalg.norm(x)
    P=np.eye(m-i)-2*(v@np.transpose(v))/(np.transpose(v)@v)
    
    H=np.block([[I, np.zeros((i, m-i))], 
                [np.zeros((m-i, i)), P]])
    return H

# 2.3) Descomposició QR en operacions sobre la matriu A i el vector b: 
    
def QRHouseholder(A, b):
    m,n=A.shape
    for k in range(n):
        x=A[k:m, k]
        vk=np.copy(x).reshape(len(x), 1)
        vk[0]=vk[0]+np.sign(x[0])*np.linalg.norm(x)
        vk=vk/np.linalg.norm(vk)
        A[k:m, k:n]=A[k:m, k:n]-2*vk@(np.transpose(vk)@(A[k:m, k:n]))
        b[k:m]=b[k:m] -2*vk@(np.transpose(vk)@(b[k:m]))
    return A, b