import jax.numpy as jnp
import jax

class ClassicalController:
    def __init__(self, gains):
        self.gains = jnp.array(gains)  # [K_p, K_i, K_d]
        self._error_history = jnp.array([]) 

    def _calc_proportional(self):
        return self.gains[0] * ( self._error_history[-1] if self._error_history.size > 0 else jnp.array(0.0) )

    def _calc_integral(self):
        return self.gains[1] * ( jnp.sum(self._error_history) if self._error_history.size > 0 else jnp.array(0.0) )

    def _calc_derivative(self):
        if self._error_history.size < 2:
            return jnp.array(0.0)
        return self.gains[2] * (self._error_history[-1] - self._error_history[-2])

    def get_control_signal(self):
        return self._calc_proportional() + self._calc_integral() + self._calc_derivative()

    # Returns control signal U
    def step(self, error_history, error):
        # jax follows the functional paragim 
        self._error_history = jnp.concatenate([error_history, jnp.array([error])])
        return self._error_history, self.get_control_signal()
