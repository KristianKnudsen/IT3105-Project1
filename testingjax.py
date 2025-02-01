import math

def debug_simulate(p1, p2, p3, h_init=0.5, n_times=10):
    """
    Pure Python simulation of the environment with PID parameters p1, p2, p3.
    This won't be traced by JAX, so we can safely print or store anything we want.
    """
    h = h_init
    errors = []
    heights = []

    for step in range(n_times):
        # Current error is target (0.5) - current height
        e = 0.5 - h
        errors.append(e)

        # Derivative term
        if step > 0:
            e_deriv = e - errors[-2]
        else:
            e_deriv = 0.0

        # PID Control
        U = p1 * e + p2 * sum(errors) + p3 * e_deriv

        # Environment update (exact copy of your get_error logic, but Pythonic)
        a = 1.5
        c = 0.015
        vc = math.sqrt(19.6 * h) * c
        hd = (U - vc) / a
        h = max(0, h + hd)

        heights.append(h)

        # Print debug info
        print(f"Step {step}: e={e:.4f}, U={U:.4f}, new_height={h:.4f}")

    return heights

# Your training code
# ---------------------------------------------------------------------
import jax.numpy as jnp
from jax import grad

def get_control(p1, p2, p3, errors, step):
    kp = p1 * errors[step - 1] if step > 0 else 0.0
    ki = p2 * jnp.sum(errors[:step]) if step > 0 else 0.0
    kd = p3 * (errors[step - 1] - errors[step - 2]) if step > 1 else 0.0
    return kp + ki + kd

def get_error(U, h):
    a = 1.5
    c = 0.015
    vc = jnp.sqrt(19.6 * h) * c
    hd = (U - vc) / a
    return jnp.maximum(0, h + hd)

def compute_avg_error(p1, p2, p3, h, n_times=10):
    errors = jnp.zeros(n_times)
    for step in range(n_times):
        U = get_control(p1, p2, p3, errors, step)
        h = get_error(U, h)
        errors = errors.at[step].set(0.5 - h)
    return jnp.mean(errors)

# Initial p.rameters
param1 = 0.1
param2 = 0.1
param3 = 0.1
h_init = 0.5
n_times = 20

lr = 0.1
epochs = 20

print("Initial parameters:", param1, param2, param3)

for epoch in range(epochs):
    grad_fn = grad(compute_avg_error, argnums=(0, 1, 2))
    grad_p1, grad_p2, grad_p3 = grad_fn(param1, param2, param3, h_init, n_times)
    param1 -= lr * grad_p1
    param2 -= lr * grad_p2
    param3 -= lr * grad_p3

    print(f"\nEpoch {epoch+1}")
    print(f"Grad w.r.t p1: {grad_p1:.6f}, p2: {grad_p2:.6f}, p3: {grad_p3:.6f}")
    print(f"Updated params: p1={param1:.4f}, p2={param2:.4f}, p3={param3:.4f}")

    # Run pure-Python debug simulation with the latest params
    print("Debug Simulation:")
    debug_simulate(param1, param2, param3, h_init, n_times)
    print("-" * 40)

print("\nFinal PID params:", param1, param2, param3)
