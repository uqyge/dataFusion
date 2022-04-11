#%%
import collections

import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import kl_div, ndtr


def ecdf(exp):
    exp_ou = sorted(set(exp))
    freq = [collections.Counter(exp)[i] for i in exp_ou]
    # print(f"freq=")
    cdf = np.cumsum(freq) / sum(freq)
    # print(f"cdf=")
    return dict(zip(exp_ou, cdf))


def kde_cdf(kde, x):
    return np.array(
        [ndtr(np.ravel(item - kde.dataset) / kde.factor).mean() for item in x]
    )


class f:
    def __init__(self, params):
        mu = params[0]
        sigma = params[1]
        self.p = stats.norm(mu, sigma)
        self.t = stats.uniform(64, 66 - 64)
        self.r = stats.uniform(8.48, 8.52 - 8.48)

        samples = 100_000
        data = np.asarray(
            [
                self.p.rvs(samples, random_state=1),
                self.t.rvs(samples, random_state=1),
                self.r.rvs(samples, random_state=1),
            ]
        ).T

        y = jax.vmap(self.model)(data)
        kde = stats.gaussian_kde(y)
        self.kde = kde

    # def model(self, p, t, r):
    def model(self, params):
        p, t, r = params
        E = 128e6
        nu = 0.24839
        c = 0.45

        p = abs(
            p * 6894.757
        )  # prevent getting a negative p, which could break sqrt in sigma.
        # p = max(0, p * 6894.757)
        t = t * 1e-6
        r = r * 1e-3

        sigma = ((-(2**0.5) / 9) * E * (t / r)) * (
            1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
        ) ** 0.5

        out = p * np.pi * r**2 - 2 * np.pi * t * r * sigma
        return c * out

    def pdf(self, x):
        return self.kde.pdf(x)

    def cdf(self, x):
        return kde_cdf(self.kde, x)


if __name__ == "__main__":
    #%%
    mu = 20
    sigma = 2
    print(f"{f([mu, sigma]).model([20, 64, 8.48])=}")
    x = np.linspace(0, 40, 100)
    plt.plot(x, f([mu, sigma]).pdf(x), label="pdf")
    plt.plot(x, f([mu, sigma]).cdf(x), label="cdf")
    plt.legend()

# %%
