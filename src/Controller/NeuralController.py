# NeuralController.py
import jax
import jax.numpy as jnp

class NeuralController:
    def __init__(self, param_array, hidden_size=16):
        self._error_history = jnp.array([])
        self.hidden_size = hidden_size
        self.params = param_array  # flattened array of NN parameters

    def _unpack_params(self):
        # Unpack the 1D param_array into W1, b1, W2, b2
        input_dim = 3
        output_dim = 1
        size_W1 = input_dim * self.hidden_size
        size_b1 = self.hidden_size
        size_W2 = self.hidden_size * output_dim
        size_b2 = output_dim
        offset = 0
        W1 = self.params[offset : offset + size_W1].reshape((input_dim, self.hidden_size))
        offset += size_W1
        b1 = self.params[offset : offset + size_b1]
        offset += size_b1
        W2 = self.params[offset : offset + size_W2].reshape((self.hidden_size, output_dim))
        offset += size_W2
        b2 = self.params[offset : offset + size_b2]
        return W1, b1, W2, b2

    def _calc_proportional(self):
        return self._error_history[-1] if self._error_history.size > 0 else 0.0

    def _calc_integral(self):
        return jnp.sum(self._error_history) if self._error_history.size > 0 else 0.0

    def _calc_derivative(self):
        if self._error_history.size < 2:
            return 0.0
        return self._error_history[-1] - self._error_history[-2]

    def get_control_signal(self):
        W1, b1, W2, b2 = self._unpack_params()
        p = self._calc_proportional()
        i = self._calc_integral()
        d = self._calc_derivative()
        x = jnp.array([p, i, d])
        x = jax.nn.relu(x @ W1 + b1)
        x = x @ W2 + b2
        return x[0]

    def step(self, error_history, error):
        self._error_history = jnp.concatenate([error_history, jnp.array([error])])
        return self._error_history, self.get_control_signal()
