import jax.numpy as jnp
 
class CournotPlant:
    def __init__(self, p_max: float, c_m: float) -> None:
        """
        Arguments:
        p_max : float
            The maximum price.
        c_m : float
            The marginal cost.
        """
        self.p_max = jnp.array(p_max)
        self.c_m = jnp.array(c_m)

    def error(self, Y: jnp.ndarray, T):
        """
        Arguments:
        Y : jnp.ndarray
            A tuple of two elements (q1, q2) representing the quantities.
        T : jnp.array
            The target profit.
        
        Returns:
        jnp.array
            The difference (T - profit1).
        """
        q_total = jnp.sum(Y)
        p = self.p_max - q_total
        profit1 = Y[0] * (p - self.c_m)
        return T - profit1

    def step(self, Y: jnp.ndarray, U, D) -> jnp.ndarray:
        """
        Arguments:
        Y : jnp.ndarray
            The current state as a tuple of quantities (q1, q2).
        U : jnp.ndarray
            The control action for the first quantity.
        D : floa
            The disturbance or control action for the second quantity.
        
        Returns:
        jnp.ndarray
            The updated state (q1, q2) after applying the actions and clipping.
        """
        Y_new = jnp.clip(Y + jnp.array([U, D]), 0, 1)
        return Y_new