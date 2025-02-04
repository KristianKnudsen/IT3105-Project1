import jax.numpy as jnp
import jax
from Plants.PlantInterface import Plant
 
class PendulumPlant(Plant):
    def __init__(self, C_Drag, Area, mass, Voltage):

        # Density of air, we assume this is generally not subject to change
        p = 1.225
        self.C =  jnp.array( C_Drag * Area * p / 2 )
        self.mass = jnp.array( mass )
        self.volt = jnp.array(Voltage)
        
    def error(self, Y, T):
        return T - Y

    def step(self, Y, U, D):
        """
        Y: Current state
        U: Controller Input, In this case amps
        D: Disturbance, directly applied to the state
        """

        v_max = jnp.sqrt(2 * Y / self.mass)
        v_avg = (2 / jnp.pi) * v_max
        d_E = self.C * v_avg**3
        E_i = self.volt * U
        E_new = Y + E_i - d_E + D
        return jnp.maximum(E_new, 1e-8)
