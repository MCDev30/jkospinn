from pkg import *


class DoubleWellProcess:
    def __init__(self, alpha: float = 1.0, sigma: float = 0.8):
        self.alpha = alpha
        self.sigma = sigma
        self.name = "DoubleWell"

    def potential(self, x):
        return self.alpha * (x**2 - 1)**2

    def drift(self, x):
        return -4 * self.alpha * x * (x**2 - 1)

    def get_true_params(self):
        return {"alpha": self.alpha, "sigma": self.sigma}

    def stationary_density(self, x):
        D = self.sigma**2
        potential = self.potential(x)
        unnormalized = jnp.exp(-potential / D)
        x_grid = jnp.linspace(-4, 4, 200)
        Z = jnp.trapz(jnp.exp(-self.potential(x_grid) / D), x_grid)
        return unnormalized / Z
