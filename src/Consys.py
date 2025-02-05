from Controller.ControllerBase import ControllerBase
from Plants.PlantInterface import Plant
import jax
import jax.numpy as jnp
import jax.random as jr

class Consys:
    def __init__(
        self,
        controller: ControllerBase,
        plant: Plant,
        initial_state,
        setpoint,
        time_steps,
        disturbance_range,
        seed=1337
    ):
        """
        Initializes the simulation with:
          - controller: a predefined controller instance
          - plant: an instance of a Plant (e.g., BathtubPlant, CournotPlant, etc.)
          - initial_state: initial state (e.g., water height or production quantity)
          - setpoint: target setpoint (scalar or vector, depending on the plant)
          - time_steps: number of simulation steps
          - disturbance_range: (min, max) range for random disturbances
          - seed: reproducibility and jax shenanigans
        """
        self.controller = controller
        self.plant = plant
        self.setpoint = jnp.array(setpoint)
        self.time_steps = time_steps
        self.key = jr.PRNGKey(seed)  # JAX PRNG key
        self.disturbance_range = disturbance_range
        self.state = jnp.array(initial_state)

    def simulate(self):
        """
        Runs the simulation for a fixed number of time steps.

        At each time step:
          - Compute the error between the current state and the setpoint.
          - Update the controller to obtain the control signal.
          - Sample a random disturbance.
          - Update the plant state using the control signal and disturbance.

        Returns:
        float
            Mean squared error over all time steps.
        """
        error_history = jnp.array([])
        Y = self.state  # Initial state

        for _ in range(self.time_steps):
            error = self.plant.error(Y, self.setpoint)
            error_history, U = self.controller.step(error_history, error)

            self.key, subkey = jr.split(self.key)
            D = jr.uniform(
                subkey, 
                shape=(), 
                minval=self.disturbance_range[0], 
                maxval=self.disturbance_range[1]
            )

            Y = self.plant.step(Y, U, D)

        mse_loss = jnp.mean(error_history**2)
        return mse_loss

    def loss_fn(self, params):
        """
        Arguments:
        params : jnp.array
            Controller parameters to be optimized.

        Returns:
        float
            Mean squared error loss from the simulation.
        """
        self.controller.params = params
        return self.simulate()

    def train(self, epochs, learning_rate):
        """
        Uses gradient descent to optimize the controller parameters.
        
        Arguments:
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate applied to gradient descent.

        Returns:
        jnp.array
            Optimized controller parameters.
        """
        value_and_grad_fn = jax.value_and_grad(self.loss_fn)
        params = self.controller.params

        for i in range(epochs):
            loss, grads = value_and_grad_fn(params)
            params = jax.tree_map(lambda p, g: p - learning_rate*g, params, grads)
            print(f"Epoch: {i+1} | Loss: {loss:.6f}")

        # Update final gains in the controller
        self.controller.params = params
        return params
