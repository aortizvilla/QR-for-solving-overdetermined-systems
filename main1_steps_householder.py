import numpy as np
from scipy.linalg import qr
from Householder import *

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# 2.1) Exemple i comprovació de la factorització QR amb una matriu A qualsevol:

m = input('Nombre files:') 
m = int(m)
n = input('Nombre columnes:') 
n = int(n)
s = round(np.random.uniform(0,100))
rg = np.random.default_rng(s)
A = rg.random((m,n))

m, n= A.shape
A0 =A.copy()
R=A.copy()
Q=np.eye(m)
for k in range(min(m-1, n)):
    H=matriuHouseholder(R, k)
    R=H@R
    Q=H@Q
    
QR0 = np.transpose(Q)@R
Q1, R1= qr(A0)
QR1 = Q1@R1
soniguals= abs(Q1-Q).all()<1e-14
print("Comprovació Q Ok=", soniguals)
soniguals= abs(R1-R).all()<1e-14
print("Comprovació R Ok=", soniguals)
soniguals= abs(QR1-QR0).all()<1e-14
print("Comprovació QR Ok=", soniguals)


# 2.2) Factorització QR d'una A pas per pas: 
A= np.array([[1, 2, 3], [4, 5, 6], [7, 8, 7], [4, 2, 3], [4, 2, 2]], dtype=float)

A1=A.copy()
m, n= A.shape
print('Matriu original de dimensions %d x %d: \n' %(m,n), A)
R=A.copy()
name='A'
Q=np.eye(m)
for k in range(min(m-1, n)):
    H=matriuHouseholder(R, k)
    print("Matriu de Householder %d d'odre %d x %d: \n" %(k+1 ,m, m), H)
    R=H@R
    tol = 1e-14
    R[np.abs(R) < tol] = 0
    name=("H%d" %(k+1)).translate(SUB) + name
    print("Factorització de la matriu %s: \n" %name, R)
    Q=H@Q

Q= np.transpose(Q)
print("La matriu Q és: \n", Q)

#Comprovació:
Q1, R1= qr(A1)
soniguals= abs(Q1-Q).all()<1e-14
print("Comprovació Q Ok=", soniguals)
soniguals= abs(R1-R).all()<1e-14
print("Comprovació R Ok=", soniguals)

QR1 = Q1@R1
QR= Q@R
soniguals= abs(QR1-QR).all()<1e-14
print("Comprovació QR Ok=", soniguals)
soniguals= abs(QR- A1).all()<1e-14
print("Comprovació factorització QR=", soniguals)

