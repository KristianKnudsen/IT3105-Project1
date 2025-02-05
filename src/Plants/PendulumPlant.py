import jax.numpy as jnp
from Plants.PlantInterface import Plant
 
class PendulumPlant(Plant):
    def __init__(self, C_Drag, Area, mass, Voltage):
        """
        Arguments:
        C_Drag : float
            Drag coefficient of the pendulum.
        Area : float
            Cross-sectional area exposed to air.
        mass : float
            Mass of the pendulum.
        Voltage : float
            Voltage value used to convert current input to energy.
        """
        # Density of air, we assume this is generally not subject to change
        p = 1.225
        self.C =  jnp.array( C_Drag * Area * p / 2 )
        self.mass = jnp.array( mass )
        self.volt = jnp.array(Voltage)
        
    def error(self, Y, T):
        return T - Y

    def step(self, Y, U, D):
        """
        Arguments:
        Y : jnp.array
            The current energy state (E).
        U : jnp.array
            Controller input (in amps) contributing to energy increase.
        D : jnp.array
            Disturbance directly affecting the energy state.
        
        Returns:
        jnp.array
            The updated energy state, ensuring it remains above a minimal safe level (for differentiation).
        """
        v_max = jnp.sqrt(2 * Y / self.mass)
        v_avg = (2 / jnp.pi) * v_max
        d_E = self.C * v_avg**3
        E_i = self.volt * U
        E_new = Y + E_i - d_E + D
        return jnp.maximum(E_new, 1e-8)
