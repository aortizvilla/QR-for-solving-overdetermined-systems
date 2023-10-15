import  numpy as np
import matplotlib.pyplot as plt
from Householder import *

data = np.genfromtxt("1880-2020.csv", delimiter = ",", skip_header = 5)
year = data[:,0]
temp = data[:,1]

# 3.1) Sistema d'equacions normals:
# Mètode 1: 
        
A=np.zeros((len(year), 6))
# Generem la matriu:
for i in range(len(year)):
    for j in range(6):
        A[i,j]=year[i]**j
        
b = temp.copy()

condicio= np.linalg.cond(A.T@A)
print("El número de condició de la matriu és:", condicio)
c1=np.linalg.solve(A.T@A, A.T@b)
print("Solucio del sistema per mínims quadrats:\n", c1.reshape(6,1))

x1=A@c1
print("El residu és:", np.linalg.norm(x1-b))


# Mètode 2: construcció de la matriu Vandermonde: 
b = temp.copy()
B= np.flip(np.vander(year, 6), 1)
condicio= np.linalg.cond(B.T@B)
print("El número de condició de la matriu és:", condicio)
c2=np.linalg.solve(B.T@B, B.T@b)
print("Solucio del sistema per mínims quadrats:", c2.reshape(6,1))

x2=A@c2
print("El residu és:", np.linalg.norm(x2-b))
print("Diferència entre l'estimació dels coeficients pels dos mètodes:\n", np.linalg.norm(c1-c2))


# 3.2) Resolució sistema lineal codi QR

A0= A.copy()
b0 = b.copy()

Aqr, bqr = QRHouseholder(A0, b0)
tol = 1e-14
Aqr[np.abs(Aqr) < tol] = 0

m, n= Aqr.shape
Aqr_mod = Aqr[:n, :n]
bqr_mod = bqr[:n]

y0=np.linalg.solve(Aqr_mod,bqr_mod)
print("Solució del sistema mitjançant la descomposició QR:\n", y0.reshape(6,1))
y1=A@y0
print("El residu és:", np.linalg.norm(y1-b))


v=np.zeros(len(year))
for i in range(len(year)): 
   v[i]=np.polyval(np.flip(c1), year[i])
   
u=np.zeros(len(year))
for i in range(len(year)): 
   u[i]=np.polyval(np.flip(c2), year[i])
   
print("Diferència avaluació polinomi entre els dos mètodes\n d'equacions normals:", np.linalg.norm(u-v))
   
w=np.zeros(len(year))
for i in range(len(year)): 
   w[i]=np.polyval(np.flip(y0), year[i])
   
print("Diferència avaluació polinomi entre els dos mètodes:\n", np.linalg.norm(u-w))

plt.figure()
plt.scatter(year, data[:,1], s=4)
plt.plot(year, v, label="Polinomi aproximat per equacions normals", color='red')
plt.plot(year, w, label="Polinomi aproximat per QR", color='green')
plt.ylabel("Anomalies tèrmiques globals (ºC)")
plt.xlabel("Anys") 
plt.legend()
plt.grid(linestyle='-', linewidth=0.2)
plt.show()

#3.5)
print("Predicció anomalia tèrmica al 2030 per l'aproximació d'equacions normals:", np.polyval(np.flip(c1), 2030))
print("Predicció anomalia tèrmica al 2030 per l'aproximació QR:", np.polyval(np.flip(y0), 2030))

