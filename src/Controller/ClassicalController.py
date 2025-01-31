import jax.numpy as jnp

class ClassicalController:
    def __init__(self, k_p, k_i, k_d):
        self.k_p = jnp.array(k_p)
        self.k_i = jnp.array(k_i)
        self.k_d = jnp.array(k_d)
        self.error_history = jnp.array([]) 

    def calc_proportional(self):
        return self.k_p * self.error_history[-1] if self.error_history.size > 0 else jnp.array(0.0)

    def calc_integral(self):
        return self.k_i * jnp.sum(self.error_history) if self.error_history.size > 0 else jnp.array(0.0)

    def calc_derivative(self):
        if self.error_history.size < 2:
            return jnp.array(0.0)
        return self.k_d * (self.error_history[-1] - self.error_history[-2])

    def get_control_signal(self):
        return self.calc_proportional() + self.calc_integral() + self.calc_derivative()

    def step(self, error):
        self.error_history = jnp.concatenate([self.error_history, jnp.array([error])])
        return self.get_control_signal()    