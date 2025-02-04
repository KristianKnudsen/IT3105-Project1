# NeuralController.py
import jax
import jax.numpy as jnp
from Controller.ControllerBase import ControllerBase

class NeuralController(ControllerBase):
    def __init__(self, params, seed=1337):
        super().__init__()
        self.params = params
        self.seed = seed

    def _init_mlp(layers)

    # Forward pass
    def get_control_signal(self):
        # Build a 3D input vector from base-class features
        x = jnp.array([
            self._calc_proportional(),  # current error
            self._calc_integral(),      # sum of errors
            self._calc_derivative()     # error difference
        ])

        # Suppose self.params is shape (4,)
        # first 3 are W, last is bias
        W = self.params[:3]  # shape (3,)
        b = self.params[3]   # scalar

        # Single-layer linear pass: U = x·W + b
        U = jnp.dot(x, W) + b
        return U