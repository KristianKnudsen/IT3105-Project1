from BathtubPlant import BathtubPlant
import random as rd

U = 0.04

H = 0.5

def getD(min=-0.01, max=0.01):
    return rd.uniform(-0.01, 0.01)

bp = BathtubPlant(H_0=H, A=1.5, C=0.015)

epochs = 10
for _ in range(epochs):
    bp.iterate(U=U, D=getD())
    print(bp.H)