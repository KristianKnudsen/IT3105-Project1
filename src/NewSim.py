# PIDSimulation.py
import jax
import jax.numpy as jnp
import jax.random as jr
from Controller.ClassicalController import ClassicalController
from Plants.BathtubPlant import BathtubPlant

def run_simulation(gains, steps=100, seed=42, setpoint=0.5):
    cc = ClassicalController(gains)
    # Now we can do 3-arg init with (H_0, A, C)
    bp = BathtubPlant(0.5, 0.15, 0.15/100)
    key = jr.PRNGKey(seed)
    H = bp.initial_H  # This is 0.5
    for _ in range(steps):
        e = setpoint - H
        error_history, u = cc.step(cc._error_history, e)
        key, subkey = jr.split(key)
        d = jr.uniform(subkey, minval=-0.01, maxval=0.01)
        H = bp.step(H, u, d)
    return jnp.mean(cc._error_history**2)

@jax.jit
def loss_fn(gains):
    return run_simulation(gains)

def main():
    lr = 0.1
    iters = 100
    gains = jnp.array([0.1, 0., 0.])
    grad_fn = jax.grad(loss_fn)
    for i in range(iters):
        print(i)
        g = grad_fn(gains)
        gains = gains - lr * g
        if i % 10 == 0:
            print(i, loss_fn(gains), gains)
    print("Optimized gains:", gains, "Loss:", loss_fn(gains))

if __name__ == "__main__":
    main()
