import jax.numpy as jnp
import jax
 
class CournotPlant:
    # State Y should be a tuple of 2 elements q1 and q2.
    def __init__(self, p_max: float, c_m: float) -> None:
        self.p_max = jnp.array(p_max)
        self.c_m = jnp.array(c_m)

    def error(self, Y: jnp.ndarray, T: float) -> float:
        q_total = jnp.sum(Y)
        p = self.p_max - q_total
        profit1 = Y[0] * (p - self.c_m)
        return T - profit1

    def step(self, Y: jnp.ndarray, U: float, D: float) -> jnp.ndarray:
        Y_new = jnp.clip(Y + jnp.array([U, D]), 0, 1)
        return Y_new