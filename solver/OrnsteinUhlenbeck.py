from pkg import *
from config import *


class OUProcess:
    def __init__(self, theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.name = "OU"

    def drift(self, x):
        return -self.theta * (x - self.mu)

    def get_true_params(self):
        return {"theta": self.theta, "mu": self.mu, "sigma": self.sigma}

    def stationary_density(self, x):
        variance = self.sigma**2 / (2 * self.theta)
        return stats.norm.pdf(x, loc=self.mu, scale=jnp.sqrt(variance))
