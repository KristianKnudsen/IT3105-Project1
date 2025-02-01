import jax
import jax.numpy as jnp
import jax.random as jr

class NeuralController:
    def __init__(self, rng, hidden_size=16):
        self._error_history = jnp.array([])
        k1, k2, k3, k4 = jr.split(rng, 4)
        self.params = {
            "W1": jr.normal(k1, (3, hidden_size)) * 0.1,
            "b1": jr.normal(k2, (hidden_size,)) * 0.1,
            "W2": jr.normal(k3, (hidden_size, 1)) * 0.1,
            "b2": jr.normal(k4, (1,)) * 0.1
        }

    def _calc_proportional(self):
        return self._error_history[-1] if self._error_history.size > 0 else 0.0

    def _calc_integral(self):
        return jnp.sum(self._error_history) if self._error_history.size > 0 else 0.0

    def _calc_derivative(self):
        if self._error_history.size < 2:
            return 0.0
        return self._error_history[-1] - self._error_history[-2]

    def _forward(self, p, i, d):
        x = jnp.array([p, i, d])
        x = x @ self.params["W1"] + self.params["b1"]
        x = jax.nn.relu(x)
        x = x @ self.params["W2"] + self.params["b2"]
        return x[0]

    def get_control_signal(self):
        p = self._calc_proportional()
        i = self._calc_integral()
        d = self._calc_derivative()
        return self._forward(p, i, d)

    def step(self, error_history, error):
        self._error_history = jnp.concatenate([error_history, jnp.array([error])])
        return self._error_history, self.get_control_signal()
