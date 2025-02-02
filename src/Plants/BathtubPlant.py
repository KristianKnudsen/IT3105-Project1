import jax.numpy as jnp
import jax
from Plants.PlantInterface import Plant
 
class BathtubPlant(Plant):
    """
        This is a simulation of a bathtub and its water height.

        We keep track of the current water height and change it based on input flow and output flow. 
        Each timestep is assumed to be precisely 1 second.

        The input flow is denoted by U and is provided at every timestep by a controller. You can imagine 
        this as the valve of the bathtub with the unit being the volume of water per second. 
        Additionally, there's the disturbance D, given at every timestep.

        The output is a static function with the water height as its variable. The amount of water the system
        loses each timestep can be calculated based on the relative difference between the cross-sectional area
        of the drain and the bathtub, and the height the water is at. 

        Constant parameters for this class include:
        - A: Cross-section of area of the bathtub m^2
        - C: Cross-section of area of the drain m^2

        Variables for this class:
        - H: Water height m
        - U: Input flow rate m^3 / s
        - D: Disturbance to input flow rate m^3 / s

        # For autograd all the variables need to be in jnp format.
    """
    
    def __init__(self, A: float, C: float) -> None:
        self.A = jnp.array(A)
        self.C = jnp.array(C)

    # Target height - actual height
    def error(self, Y, T):
        return T - Y

    # returns new height
    def step(self, Y, U, D):
        # The function is not differentiblabla at h=0, so we simulate replace with very low water level.
        H_safe = jnp.maximum(Y, 1e-8)
        vc = jnp.sqrt(19.6 * H_safe) * self.C
            
        d_h = (U + D - vc) / self.A
        H_new = Y + d_h
        # While, returning a height of 0 at this might not cause an issue, it's best to be safe.
        return jnp.maximum(H_new, 1e-8)