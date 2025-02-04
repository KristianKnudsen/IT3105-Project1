from abc import ABC, abstractmethod

class Plant(ABC):
    @abstractmethod
    def step(self, Y: object, U: float, D: float) -> object: # State, Control param, Disturbance
        """Abstract method to update plant state.
        
        Y: Current state.
        U: Controller Input
        D: Disturbance

        returns a new state Y
        """
        pass

    @abstractmethod
    def error(self, Y: object, T: object) -> float: # State, Target
        """Abstract method to compute error between state and target.
        
        Y: State
        T: Target variable

        returns an error metric based on the plant
        """
        pass