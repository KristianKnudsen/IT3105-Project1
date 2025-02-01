from Controller.ClassicalController import ClassicalController
from Plants.BathtubPlant import BathtubPlant
import jax.numpy as jnp
import jax.random as jr


cc = ClassicalController([0.1, 0.01, 0.001])

errors = [0.3, 0.2, 0.1, 0., 0., 0.]

H = 0.5   # Initial water height

a = 0.15
c = a/100

bp = BathtubPlant(H, a, c)

e = 0.

key = jr.key(0)

for _ in range(100):
    print("---")
    print("Error e")
    print(e)
    u = cc.step(e)
    print("Controller output u", u)
    key, subkey = jr.split(key)
    D = jr.uniform(subkey, shape=(), minval=-0.01, maxval=0.01)  # Random D in [-0.01, 0.01]
    h = bp.step(u, D=D)
    print("new height", h)
    e = 0.5-h

cc.error_history