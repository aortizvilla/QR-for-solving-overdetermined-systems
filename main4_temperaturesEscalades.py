import  numpy as np
import matplotlib.pyplot as plt
from Householder import *

data = np.genfromtxt("1880-2020.csv", delimiter = ",", skip_header = 5)
year = data[:,0]
temp = data[:,1]

m=len(year)
year0=year[0]
yearm=year[m-1]

# 3.1) Sistema d'equacions normals:
    
A=np.zeros((m, 6))
for i in range(m):
    year[i]=(year[i]-year0)/(year[m-1]-year0)
    for j in range(6):
        A[i,j]=year[i]**j
        
b = temp.copy()

condicio= np.linalg.cond(np.transpose(A)@A)
print("El número de condició de la matriu és:", condicio)
c=np.linalg.solve(np.transpose(A)@A, np.transpose(A)@b)
print("Solucio del sistema per mínims quadrats:\n", c.reshape(6,1))

x1=A@c
print("El residu és:", np.linalg.norm(x1-b))

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

v=np.zeros(m)
for i in range(m): 
   v[i]=np.polyval(np.flip(c), year[i])
      
w=np.zeros(len(year))
for i in range(len(year)): 
   w[i]=np.polyval(np.flip(y0), year[i])
 
print("Diferència avaluació polinomi entre els dos mètodes:\n", np.linalg.norm(v-w))

plt.figure()
plt.scatter(year, data[:,1], s=4)
plt.plot(year, w, label="Polinomi aproximat per QR", color='green',  linewidth=4)
plt.plot(year, v, label="Polinomi aproximat per equacions normals", color='red')
plt.ylabel("Anomalies tèrmiques globals (ºC)")
plt.xlabel("Anys") 
plt.legend()
plt.grid(linestyle='-', linewidth=0.2)
plt.show()

#3.5)
print("Predicció anomalia tèrmica al 2030 per l'aproximació d'equacions normals:", np.polyval(np.flip(c), (2030-year0)/(yearm-year0)))
print("Predicció anomalia tèrmica al 2030 per l'aproximació QR:", np.polyval(np.flip(y0), (2030-year0)/(yearm-year0)))

