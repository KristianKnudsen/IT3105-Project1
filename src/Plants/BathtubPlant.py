import jax.numpy as jnp
 
class BathtubPlant:
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

    # returns new height
    def step(self, H, U: float, D: float) -> jnp.ndarray:
        vc = jnp.sqrt(19.6 * H) * self.C
        d_h = (U + D - vc) / self.A
        return jnp.maximum(0, H + d_h)