# NeuralController.py
import jax
import jax.numpy as jnp
from Controller.ControllerBase import ControllerBase

class NeuralController(ControllerBase):
    def __init__(self, layers, activations, seed=1337):
        super().__init__()
        layers = [3] + layers + [1]
        self.activations = []
        self.set_activations(activations)
        self.params = self.init_weights(layers)
        self.seed = seed

    def set_activations(self, activations):
        for a in activations:
            v = lambda x: x
            if a == "sigmoid":
                v = jax.nn.sigmoid
            elif a == "tanh":
                v = jax.nn.tanh
            elif a == "relu":
                v = jax.nn.relu
            self.activations.append(v)

    def init_weights(self, layer_widths, parent_key=jax.random.PRNGKey(1337), scale=0.01):
        params = []
        keys = jax.random.split(parent_key, num=len(layer_widths)-1)

        for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
            weight_key, bias_key = jax.random.split(key)
            params.append([
                        scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                        scale*jax.random.normal(bias_key, shape=(out_width,))
                        ]
            )

        return params

    # Forward pass
    def get_control_signal(self):
        # Build a 3D input vector from base-class features
        x = jnp.array([
            self._calc_proportional(),  # current error
            self._calc_integral(),      # sum of errors
            self._calc_derivative()     # error difference
        ])

        for (w, b), act in zip(self.params, self.activations):
            x = jnp.dot(w, x) + b
            x = act(x) 

        return x.squeeze()