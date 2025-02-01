from Controller.ClassicalController import ClassicalController
from Plants.BathtubPlant import BathtubPlant
import jax
import jax.numpy as jnp
import jax.random as jr

class PIDSimulation:
    def __init__(self, gains, plant, setpoint=0.5, time_steps=100, seed=0):
        """
        Initializes the simulation with:
          gains: PID gains [K_p, K_i, K_d]
          plant: an instance of BathtubPlant
          setpoint: the target water height
          time_steps: how many steps to simulate
          seed: PRNG seed for reproducibility
        """
        self.controller = ClassicalController(gains)  # create once here
        self.plant = plant
        self.setpoint = jnp.array(setpoint)
        self.time_steps = time_steps
        self.key = jr.PRNGKey(seed)  # JAX PRNG for reproducibility

    def simulate(self):
        """
        Runs the simulation for `self.time_steps`.
        Returns:
          - error_history (jnp array of shape [time_steps])
          - final water height H
          - mse_loss (mean of squared errors)
        """
        error_history = jnp.array([])
        H = jnp.array(0.5)  # Initial water height

        for _ in range(self.time_steps):

            error = self.setpoint - H
            error_history, U = self.controller.step(error_history, error)

            # Random disturbance
            self.key, subkey = jr.split(self.key)  # Update PRNG key
            D = jr.uniform(subkey, shape=(), minval=-0.01, maxval=0.01)

            H = self.plant.step(H, U, D)

        mse_loss = jnp.mean(error_history**2)
        return error_history, H, mse_loss

def loss_fn(gains):
    """
    Given PID gains, run the simulation and return the MSE loss.
    We'll differentiate this w.r.t. 'gains'.
    """
    plant = BathtubPlant(A=0.15, C=0.15/100)
    sim = PIDSimulation(gains, plant, setpoint=0.5, time_steps=100, seed=42)
    _, _, mse_loss = sim.simulate()
    return mse_loss

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.1  # Step size for gradient descent
    num_iterations = 100  # Number of optimization steps

    # Initialize PID gains
    gains = jnp.array([0.1, 0., 0.])

    print(f"Initial gains: {gains}")

    for i in range(num_iterations):
        # Compute gradient of loss w.r.t. PID gains
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(gains)

        # Gradient descent update
        gains -= learning_rate * grads  # Update gains using learning rate

        # Compute new loss
        loss = loss_fn(gains)

        # Print progress every 10 iterations
        if i % 10 == 0 or i == num_iterations - 1:
            print(f"Iteration {i+1}/{num_iterations} - Loss: {loss:.6f} - Gains: {gains}")

    print("\nOptimized PID gains:", gains)