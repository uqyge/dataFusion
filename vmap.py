# %%
import jax
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import time

# def model(p, t, r):
def model(params):
    p, t, r = params
    E = 128e6
    nu = 0.24839

    p = p * 6894.757
    # p = max(0, p * 6894.757)
    t = t * 1e-6
    r = r * 1e-3

    # sigma = ((-math.sqrt(2) / 9) * E * (t / r)) * math.sqrt(
    #     1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
    # )
    sigma = ((-(2**0.5) / 9) * E * (t / r)) * (
        1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
    ) ** 0.5

    # sigma = 1
    out = p * math.pi * r**2 - 2 * math.pi * t * r * sigma
    return out * 0.5


def model2(params):
    p, t, r = params
    E = 128e6
    nu = 0.24839

    p = p * 6894.757
    # p = max(0, p * 6894.757)
    t = t * 1e-6
    r = r * 1e-3

    sigma = ((-math.sqrt(2) / 9) * E * (t / r)) * math.sqrt(
        1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
    )
    # sigma = ((-(2**0.5) / 9) * E * (t / r)) * (
    #     1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
    # ) ** 0.5

    # sigma = 1
    out = p * math.pi * r**2 - 2 * math.pi * t * r * sigma
    return out * 0.5


#%%
mu = 20
sigma = 1.5
p = stats.norm(mu, sigma)
t = stats.uniform(64, 66 - 64)
r = stats.uniform(8.48, 8.52 - 8.48)
samples = 10_000_000
data = [p.rvs(samples), t.rvs(samples), r.rvs(samples)]


#%%
# y = [model(data[0][i], data[1][i], data[2][i]) for i in range(samples)]

# %%
# x = np.linspace(10, 25, 100)
# kde = stats.gaussian_kde(y)
# plt.plot(x, kde.pdf(x))
# plt.hist(y, bins=20, density=True)

# %%
#%%

# %%
samples = 100_000
data = [p.rvs(samples), t.rvs(samples), r.rvs(samples)]
d2 = np.asarray(data).T
d2.shape


v_model = jax.vmap(model)
# %%
# %time
a = time.time()
tmp = v_model(d2)
b = time.time()
b - a

#%%
x = np.linspace(10, 25, 100)
kde = stats.gaussian_kde(tmp)
plt.plot(x, kde.pdf(x))
# plt.hist(tmp, bins=20, density=True)

# %%
a = time.time()
tmp = np.asarray([model(d2[i]) for i in range(samples)])
b = time.time()
b - a
# %%
a = time.time()
tmp2 = np.asarray([model2(d2[i]) for i in range(samples)])
b = time.time()
b - a

# %%
x = np.linspace(10, 25, 100)
kde = stats.gaussian_kde(tmp)
plt.plot(x, kde.pdf(x))


# %%
# %%
(tmp-tmp2).sum()
# %%
