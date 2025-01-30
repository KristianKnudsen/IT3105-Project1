import numpy as np
import math

class BathtubPlant:
    def __init__(self, H_0: np.float64, A: np.float64, C: np.float64 ) -> None:
        self.H = H_0
        self.A = A
        self.C = C

    def iterate(self, U: np.float64, D: np.float64) -> np.float64:
        # Ensure the bathtub doesn't go into negative water height.
        self.H = max( 0, self.H + self.get_height_delta(U, D) )
        return self.H
    
    def get_height_delta(self, U, D) -> np.float64:
        vc = math.sqrt(19.6 * self.H) * self.C
        return ( U + D - vc ) / self.A 