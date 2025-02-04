from Controller.ClassicalController import ClassicalController
from Plants.BathtubPlant import BathtubPlant
from Plants.PlantInterface import Plant
from Plants.CournotPlant import CournotPlant
from Controller.NeuralController import NeuralController
import jax
import jax.numpy as jnp
import jax.random as jr

class Consys:
    def __init__(
        self,
        controller: ClassicalController,   # Pass in a controller instance
        plant: Plant,                     # Pass in a plant instance
        initial_state,                    # Initial state Y
        setpoint,
        time_steps,
        disturbance_range,  # Disturbance range
        seed=1337
    ):
        """
        Initializes the simulation with:
          controller: a ClassicalController (PID or other)
          plant: an instance of a Plant (BathtubPlant, CournotPlant, etc.)
          initial_state: initial state (e.g. Y in the bathtub or Q in Cournot model)
          setpoint: the target setpoint (scalar or vector, depending on the plant)
          time_steps: how many steps to simulate
          disturbance_range: (min, max) range for random disturbance
          seed: PRNG seed for reproducibility
        """
        self.controller = controller
        self.plant = plant
        self.setpoint = jnp.array(setpoint)
        self.time_steps = time_steps
        self.key = jr.PRNGKey(seed)  # JAX PRNG key
        self.disturbance_range = disturbance_range
        # Store the initial state as a JAX array
        self.state = jnp.array(initial_state, dtype=jnp.float32)

    def simulate(self):
        """
        Runs the simulation for `self.time_steps` using the current controller gains.
        Returns:
            - mse_loss (float): mean of squared errors over all time steps
        """
        error_history = jnp.array([])
        Y = self.state  # Current state

        for _ in range(self.time_steps):
            error = self.plant.error(Y, self.setpoint)
            # Step the controller
            error_history, U = self.controller.step(error_history, error)

            # Random disturbance
            self.key, subkey = jr.split(self.key)
            D = jr.uniform(
                subkey, 
                shape=(), 
                minval=self.disturbance_range[0], 
                maxval=self.disturbance_range[1]
            )

            # Step the plant dynamics
            Y = self.plant.step(Y, U, D)

        mse_loss = jnp.mean(error_history**2)
        return mse_loss

    def loss_fn(self, params):
        """
        Helper method for JAX differentiation.
        Sets the controller gains, then returns the MSE from simulate().
        """
        self.controller.params = params
        return self.simulate()

    def train(self, num_iterations, learning_rate):
        """
        Uses gradient descent to optimize the PID gains.
        
        Args:
            num_iterations (int): number of training iterations
            learning_rate (float): step size for gradient update

        Returns:
            jnp.array: optimized gains
        """
        # We define a function that returns the value and gradient of loss_fn
        value_and_grad_fn = jax.value_and_grad(self.loss_fn)

        # Start from the current controller gains
        params = self.controller.params

        for i in range(num_iterations):
            loss, grads = value_and_grad_fn(params)
            params = params - learning_rate * grads
            print(f"Iteration {i+1} | Loss: {loss:.6f} | Gains: {params}")

        # Update final gains in the controller
        self.controller.params = params
        return params


if __name__ == "__main__":
    #controller = ClassicalController(params=jnp.array([0.0, 0.0, 0.0]))
    controller = NeuralController(params=jnp.zeros(4))
    plant = CournotPlant(p_max=2.0, c_m=0.1)

    sim = Consys(
        controller=controller,
        plant=plant,
        initial_state=[0.0, 0.5],
        setpoint=0.3,
        time_steps=20,
        disturbance_range=(-0.0, 0.0),
        seed=42
    )

    optimized_gains = sim.train(num_iterations=50, learning_rate=0.001)
    print("\nOptimized PID gains:", optimized_gains)
