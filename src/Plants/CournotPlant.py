import jax.numpy as jnp
import jax
 
class CournotPlant:
    # State Y should be a tuple of 2 elements q1 and q2.

    def __init__(self, p_max: float, c_m: float) -> None:
        self.p_max = jnp.array(p_max)
        self.c_m = jnp.array(c_m)

    def error(self, Y: tuple, T):
        q1 = Y[0]
        q2 = Y[1]
        q = q1 + q2
        p = jnp.maximum( self.p_max - q, 0)  
        p1 = q1 * (p - self.c_m)
        return T - p1

    def step(self, Y: tuple, U: float, D: float) -> tuple:
        q1, q2 = Y
        q1_new = jnp.clip(q1 + U, 0, 1)
        q2_new = jnp.clip(q2 + D, 0, 1)

        q_total = q1_new + q2_new
        p = jnp.maximum(self.p_max - q_total, 0)
        
        return (q1_new, q2_new)