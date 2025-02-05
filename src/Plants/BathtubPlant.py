import jax.numpy as jnp
import jax
from Plants.PlantInterface import Plant
 
class BathtubPlant(Plant):
    """
    Simulation of a bathtub system where the water height is updated each second based on the input flow,
    disturbance, and outflow calculated from the current water height.
    
    Constant parameters:
    - A: Cross-sectional area of the bathtub (m^2)
    - C: Cross-sectional area of the drain (m^2)
    
    Variables:
    - H: Water height (m)
    - U: Input flow rate (m^3/s)
    - D: Disturbance to input flow (m^3/s)
    """
    
    def __init__(self, A: float, C: float) -> None:
        """
        Arguments:
        A : float
            Cross-sectional area of the bathtub (m^2).
        C : float
            Cross-sectional area of the drain (m^2).
        """
        self.A = jnp.array(A)
        self.C = jnp.array(C)

    def error(self, Y, T):
        """
        Arguments:
        Y : jnp.array
            Current water height.
        T : jnp.array
            Target water height.
        
        Returns:
        jnp.array
            The difference (T - Y).
        """
        return T - Y

    def step(self, Y, U, D):
        """
        Arguments:
        Y : jnp.array
            Current water height.
        U : jnp.array
            Input flow rate (m^3/s).
        D : jnp.array
            Disturbance to the input flow (m^3/s).
        
        Returns:
        jnp.array
            The updated water height (m).
        """      
        # The function is not differentiblabla at h=0, so we simulate replace with very low water level.
        H_safe = jnp.maximum(Y, 1e-8)
        vc = jnp.sqrt(19.6 * H_safe) * self.C
            
        d_h = (U + D - vc) / self.A
        H_new = Y + d_h
        # While, returning a height of 0 at this might not cause an issue, it's best to be safe.
        return jnp.maximum(H_new, 1e-8)