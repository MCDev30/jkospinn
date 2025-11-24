from pkg import *
from config import *

class SDESimulator:
    """Base class for SDE simulation"""

    @staticmethod
    def euler_maruyama(key, x0, drift_fn, sigma, dt, n_steps):
        def step(carry, t):
            x, subkey = carry
            subkey, subkey_noise = random.split(subkey)
            dW = random.normal(subkey_noise, shape=x.shape) * jnp.sqrt(dt)
            x_new = x + drift_fn(x) * dt + sigma * dW
            return (x_new, subkey), x_new

        keys = random.split(key, n_steps)
        _, trajectory = jax.lax.scan(step, (x0, key), jnp.arange(n_steps))
        return jnp.concatenate([x0[None], trajectory])
