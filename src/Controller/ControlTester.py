from ClassicalController import ClassicalController
import jax.numpy as jnp


cc = ClassicalController([1.0, 0.1, 0.01])

errors = [0.3, 0.2, 0.1, 0., 0., 0.]

for e in errors:
    print("---")
    print(e)
    print(cc.step(e))
