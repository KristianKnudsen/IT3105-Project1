# PIDSimulation.py
from Controller.NeuralController import NeuralController
from Plants.BathtubPlant import BathtubPlant
import jax
import jax.numpy as jnp
import jax.random as jr

class PIDSimulation:
    def __init__(self, net_params, plant, setpoint=0.5, time_steps=100, seed=0, hidden_size=16):
        self.controller = NeuralController(net_params, hidden_size)
        self.plant = plant
        self.setpoint = jnp.array(setpoint)
        self.time_steps = time_steps
        self.key = jr.PRNGKey(seed)

    def simulate(self):
        error_history = jnp.array([])
        H = jnp.array(0.5)
        for _ in range(self.time_steps):
            error = self.setpoint - H
            error_history, U = self.controller.step(error_history, error)
            self.key, subkey = jr.split(self.key)
            D = jr.uniform(subkey, shape=(), minval=-0.01, maxval=0.01)
            H = self.plant.step(H, U, D)
        mse_loss = jnp.mean(error_history**2)
        return error_history, H, mse_loss

def loss_fn(net_params):
    plant = BathtubPlant(A=0.15, C=0.15/100)
    sim = PIDSimulation(net_params, plant, setpoint=0.5, time_steps=100, seed=42, hidden_size=16)
    _, _, mse_loss = sim.simulate()
    return mse_loss

def init_net_params(rng, input_dim=3, hidden_size=16, output_dim=1):
    W1 = jax.random.normal(rng, (input_dim, hidden_size)) * 0.1
    b1 = jax.random.normal(rng, (hidden_size,)) * 0.1
    W2 = jax.random.normal(rng, (hidden_size, output_dim)) * 0.1
    b2 = jax.random.normal(rng, (output_dim,)) * 0.1
    return jnp.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

if __name__ == "__main__":
    rng = jr.PRNGKey(0)
    net_params = init_net_params(rng, input_dim=3, hidden_size=16, output_dim=1)
    learning_rate = 0.01
    num_iterations = 100
    grad_fn = jax.grad(loss_fn)

    for i in range(num_iterations):
        grads = grad_fn(net_params)
        net_params -= learning_rate * grads
        if i % 10 == 0:
            l = loss_fn(net_params)
            print(i, l, net_params[:5])
    print("Final loss:", loss_fn(net_params))
