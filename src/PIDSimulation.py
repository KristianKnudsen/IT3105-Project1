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
        if jnp.isnan(gains).any():
            jax.debug.print("NaN detected in PID gains BEFORE passing to ClassicalController: {gains}", gains=gains)
        jax.debug.print("1")
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
            jax.debug.print("3: {K_p}", K_p=U)
            jax.debug.print("g: {K_p}", K_p=self.controller.gains)

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
    sim = PIDSimulation(gains, plant, setpoint=0.5, time_steps=10, seed=42)
    _, _, mse_loss = sim.simulate()
    return mse_loss

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.001  # Step size for gradient descent
    num_iterations = 1  # Number of optimization steps

    # Initialize PID gains
    gains = jnp.array([0.5, 0., 0.])

    print(f"Initial gains: {gains}")

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    for i in range(num_iterations):
        # This single call returns both loss and gradients 
        # in one forward pass (plus the reverse-mode AD).
        loss, grads = value_and_grad_fn(gains)
        
        # Update gains
        gains -= learning_rate * grads

        # Now 'loss' is the pre-update loss, 
        # but it came from exactly the same forward pass used to compute grads.
        print(f"Iteration {i+1} - Loss: {loss:.6f} - Gains: {gains}")

    print("\nOptimized PID gains:", gains)