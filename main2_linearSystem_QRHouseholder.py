import numpy as np
from Householder import *

A= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 7], [4, 2, 3], [4, 2, 2]], dtype="float")
b=np.array([1, 1, 1, 1, 1], dtype="float")

A0=A.copy()
b0=b.copy()

# 2.3) Descomposició QR en operacions sobre la matriu A i el vector b: 

Aqr, bqr = QRHouseholder(A, b)
tol = 1e-14
Aqr[np.abs(Aqr) < tol] = 0
print("A després d'aplicar les matrius de Householder:\n", Aqr)
print("Vector utilitzant l'algorisme 'QRHouseholder':\n", bqr.reshape(5,1))

m, n= Aqr.shape
Aqr_mod = Aqr[:n, :n]
bqr_mod = bqr[:n]

x0=np.linalg.solve(Aqr_mod,bqr_mod)
print("Solució del sistema mitjançant la descomposició QR:\n", x0.reshape(3,1))
x1=A0@x0
print("A*x=", x1.reshape(5,1))
print("El residu és:", np.linalg.norm(x1-b0))




