from Plants.PendulumPlant import PendulumPlant
from Controller.NeuralController import NeuralController
from Controller.ClassicalController import ClassicalController
from Plants.BathtubPlant import BathtubPlant
from Plants.CournotPlant import CournotPlant
from Consys import Consys
import jax.numpy as jnp

if __name__ == "__main__":
    # controller = ClassicalController(params=[0.0, 0.0, 0.0])
    # controller = NeuralController(layers=[12, 4, 2], 
    #                               activations=["relu", "sigmoid", "tanh", "none"],
     #                              init_range=(0.01, 0.025))

    # plant = BathtubPlant(C=0.0015, A=0.15)
    # plant = CournotPlant(p_max=2.0, c_m=0.1)
    # plant = PendulumPlant(C_Drag=0.5, Area=0.1, mass= 1.0, Voltage= 12.0)

    sim = Consys(
        controller=controller,
        plant=plant,
        initial_state=0.5,
        setpoint=0.5,
        time_steps=20,
        disturbance_range=(-0.01, 0.01),
        seed=1337
    )

    optimized_gains = sim.train(epochs=50, learning_rate=0.00005)
