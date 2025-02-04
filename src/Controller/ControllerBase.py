from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

class ControllerBase(ABC):
    def __init__(self):
        self._error_history = jnp.array([])

    def _calc_proportional(self):
        return ( self._error_history[-1] if self._error_history.size > 0 else jnp.array(0.0) )

    def _calc_integral(self):
        return ( jnp.sum(self._error_history) if self._error_history.size > 0 else jnp.array(0.0) )

    def _calc_derivative(self):
        if self._error_history.size < 2:
            return jnp.array(0.0)
        return (self._error_history[-1] - self._error_history[-2])

    def step(self, error_history, error):
        self._error_history = jnp.concatenate([error_history, jnp.array([error])])
        U = self.get_control_signal()
        return self._error_history, U

    @abstractmethod
    def get_control_signal(self):
        raise NotImplementedError("Subclasses must implement get_control_signal()")