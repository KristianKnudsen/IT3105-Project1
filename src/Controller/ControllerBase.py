from abc import ABC, abstractmethod
import jax.numpy as jnp

class ControllerBase(ABC):
    def __init__(self):
        # MUST DEFINE SOMETHING TO OPTIMIZE self.params
        self._error_history = jnp.array([])

    def _calc_proportional(self):
        """
        Returns:
        jnp.array
            The most recent error if available, else 0.0.
        """
        return ( self._error_history[-1] if self._error_history.size > 0 else jnp.array(0.0) )

    def _calc_integral(self):
        """
        Returns:
        jnp.array
            The sum of errors if available, else 0.0.
        """
        return ( jnp.sum(self._error_history) if self._error_history.size > 0 else jnp.array(0.0) )

    def _calc_derivative(self):
        """
        Returns:
        jnp.array
            The difference between the last two errors if available, else 0.0.
        """
        if self._error_history.size < 2:
            return jnp.array(0.0)
        return (self._error_history[-1] - self._error_history[-2])

    def step(self, error_history, error):
        """
        Arguments:
        error_history : jnp.array
            The current error history.
        error : float or jnp.array (depends on the plant)
            The latest error value.

        Returns:
        tuple
            The updated error history and the computed control signal.

        Note: Jax was made with higher order programming in mind. Our solution is mostly
        based on oop which might result in some wierd interactions between the two paradigms.
        """
        self._error_history = jnp.concatenate([error_history, jnp.array([error])])
        U = self.get_control_signal()
        return self._error_history, U

    @abstractmethod
    def get_control_signal(self):
        """
        Needs to be implemented.
        Depends on the controller used.
        """
        raise NotImplementedError("Subclasses must implement get_control_signal()")