import jax.numpy as jnp
import jax
from Controller.ControllerBase import ControllerBase

class ClassicalController(ControllerBase):
    def __init__(self, params):
        super().__init__()
        self.params = params  # [K_p, K_i, K_d]

    def get_control_signal(self):
        return (
            self.params[0] * self._calc_proportional()
            + self.params[1] * self._calc_integral()
            + self.params[2] * self._calc_derivative() )