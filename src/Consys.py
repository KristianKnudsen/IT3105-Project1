from Controller.ControllerBase import ControllerBase
from Plants.PlantInterface import Plant
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

class Consys:
    def __init__(
        self,
        controller: ControllerBase,
        plant: Plant,
        initial_state,
        setpoint,
        time_steps,
        disturbance_range,
        seed,
        vizualize_loss,
        vizualize_params
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
          - loss_history: Initilizes an empty list to store loss values if True
          - params_history: Initilizes an empty list to store parameter values if True
          - param_names: List of parameter names for visualization
        """
        self.controller = controller
        self.plant = plant
        self.setpoint = jnp.array(setpoint)
        self.time_steps = time_steps
        self.key = jr.PRNGKey(seed)  # JAX PRNG key
        self.disturbance_range = disturbance_range
        self.state = jnp.array(initial_state)
        self.loss_history =  [] if vizualize_loss else None
        self.params_history = [] if vizualize_params else None
        self.param_names = vizualize_params

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
            self.store_history(loss, params)

        self.plot()
        self.controller.params = params
        return params
    
    def store_history(self, loss, params):
        """
        Stores the loss and parameter history for vizualization.

        Arguments:
        loss : float
            Current loss value.
        params : jnp.array
            Current controller parameters.
        """
        if self.loss_history is not None:
            self.loss_history.append(loss)
        if self.params_history is not None:
            self.params_history.append(params)

    def plot(self):
        """
        Plots the loss and parameter history.
        """
        if self.loss_history is not None:
            plt.figure(f"Loss_{self.controller.__class__.__name__}_{self.plant.__class__.__name__}")
            plt.plot(self.loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss history")
            plt.show()

        if self.params_history is not None:
            plt.figure(f"Params_{self.controller.__class__.__name__}_{self.plant.__class__.__name__}")
            params_history = jnp.stack(self.params_history)
            for i in range(params_history.shape[1]):
                plt.plot(params_history[:, i], label=self.param_names[i])
            plt.xlabel("Epoch")
            plt.ylabel("Parameter value")
            plt.legend(loc="upper left")
            plt.title("Parameter history")
            plt.show()
        
