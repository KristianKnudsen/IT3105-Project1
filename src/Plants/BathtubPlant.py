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

        Varying parameters for this class:
        - H: Water height m
        - U: Input flow rate m^3 / s
        - D: Disturbance to input flow rate m^3 / s

        # For autograd all the variables need to be in jnp format.
    """
    
    def __init__(self, H_0: float, A: float, C: float) -> None:
        """
        Initializes the BathtubPlant model.

        Parameters:
        - H_0: Initial water height
        - A: Cross-section of area of the bathtub
        - C: Cross-section of area of the drain
        """
        self.H = jnp.array(H_0)
        self.A = jnp.array(A)
        self.C = jnp.array(C)

    def iterate(self, U: float, D: float) -> jnp.ndarray:
        """
        Updates the water height based on control input U and disturbance D.

        Parameters:
        - U: Controller water input.
        - D: External disturbance.

        Returns:
        - Updated water height
        """
        self.H = jnp.maximum(0, self.H + self.get_height_delta(U, D))
        return self.H
    
    def get_height_delta(self, U: float, D: float) -> jnp.ndarray:
        """
        Computes the change in water height over one timestep.

        Parameters:
        - U: Controller input
        - D: Disturbance input

        Returns:
        - Change in water height (meters)
        """
        vc = jnp.sqrt(19.6 * self.H) * self.C  # JAX-compatible sqrt()
        return (U + D - vc) / self.A