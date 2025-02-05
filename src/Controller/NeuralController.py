import jax
import jax.numpy as jnp
from Controller.ControllerBase import ControllerBase

class NeuralController(ControllerBase):
    def __init__(self, layers, activations, init_range=(0.01, 0.2), seed=1337):
        """
        Arguments:
        layers : list[int]
            Defines the hidden layer sizes.
        activations : list[str]
            Names of activation functions corresponding to each hidden layer.
        init_range : tuple[float, float]
            Range for uniformly sampling weight-initialization values.
        seed : int
            Seed used for weight-initialization.
        """
        super().__init__()
        layers = [3] + layers + [1]
        self.activations = []
        self.set_activations(activations)
        self.params = self.init_weights(layers, parent_key=jax.random.PRNGKey(seed), init_range=init_range)

    def set_activations(self, activations):
        """
        Arguments:
        activations : list[str]
            Names of activation functions (e.g. 'sigmoid', 'tanh', 'relu', 'none').
        """
        for a in activations:
            v = lambda x: x
            if a == "sigmoid":
                v = jax.nn.sigmoid
            elif a == "tanh":
                v = jax.nn.tanh
            elif a == "relu":
                v = jax.nn.relu
            self.activations.append(v)

    def init_weights(self, layer_widths, parent_key, init_range):
        """
        Arguments:
        layer_widths : list[int]
            List of layer sizes.
        parent_key : jax.random.PRNGKey
            Master RNG key for splitting into per-layer RNG keys.
        init_range : tuple(float, float)
            Range (min, max) for uniformly sampling the values used for weight initialization.
        
        Returns:
        params : list of [weight, bias], so basically [[(w, w), b], ...]
            Where weight is of shape (out_width, in_width) and bias is of shape (out_width,).
        """
        params = []
        keys = jax.random.split(parent_key, num=len(layer_widths) - 1)

        for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
            weight_key, bias_key = jax.random.split(key, 2)

            weight = jax.random.uniform(
                weight_key,
                shape=(out_width, in_width),
                minval=init_range[0],
                maxval=init_range[1]
            )

            bias = jax.random.uniform(
                bias_key,
                shape=(out_width,),
                minval=init_range[0],
                maxval=init_range[1]
            )

            params.append([weight, bias])

        return params

    # Forward pass
    def get_control_signal(self):
        """
        Output is affected by the error, see ControllerBase for missing functionality

        Returns:
        jnp.array
            The scalar control signal after the final layer activation.
        """
        x = jnp.array([
            self._calc_proportional(),
            self._calc_integral(),
            self._calc_derivative()
        ])

        for (w, b), act in zip(self.params, self.activations):
            x = jnp.dot(w, x) + b
            x = act(x) 

        return x.squeeze()