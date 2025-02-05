from Controller.ControllerBase import ControllerBase
import jax.numpy as jnp

class ClassicalController(ControllerBase):
    def __init__(self, params):
        """
        Arguments:
        params : list[float]
            List of PID gains [K_p, K_i, K_d].
        """
        super().__init__()
        self.params = jnp.array(params)

    def get_control_signal(self):
        """
        Returns:
        jnp.array
            The computed control signal.
        """
        return (
            self.params[0] * self._calc_proportional()
            + self.params[1] * self._calc_integral()
            + self.params[2] * self._calc_derivative() )