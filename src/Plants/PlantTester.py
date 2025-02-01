from BathtubPlant import BathtubPlant
import random as rd

# Define initial conditions
U = 0.04  # Initial control input
H = 0.5   # Initial water height

a = 0.15
c = a/100

bp = BathtubPlant(H, a, c)

for _ in range(10):
    print(bp.step(U=0.0047, D=0))
