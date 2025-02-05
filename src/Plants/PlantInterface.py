from abc import ABC, abstractmethod

class Plant(ABC):
    @abstractmethod
    def step(self, Y: object, U, D) -> object: # State, Control param, Disturbance
        """
        Arguments:
        Y : object
            The current state.
        U : float
            Controller input.
        D : float
            Disturbance.

        Returns:
        Y : object
            The updated state.
        """
        pass

    @abstractmethod
    def error(self, Y: object, T: object): # State, Target
        """
        Arguments:
        Y : object
            The current state.
        T : object
            The target state / value.

        Returns:
        float or jnp.array
            The computed error metric.
        """
        pass