#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


#%%
from exp import ecdf

# %%
exp = [-5, -1, 0, 2, 2, 7]
exp = [i + 1 for i in exp]

#%%
exp_ou = sorted(set(exp))
y_ecdf = [ecdf(exp)[i] for i in exp_ou]

plt.step([-10, *exp_ou, 10], [0, *y_ecdf, 1], where="post")


#%%
from scipy.special import ndtr


def kde_cdf(kde, x):
    # return tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean() for item in x)
    return np.array(
        [ndtr(np.ravel(item - kde.dataset) / kde.factor).mean() for item in x]
    )


# %%
def model(x1, x2):
    return 2 * (x1 + x2)


# %%
samples = 10_000
x1 = stats.norm(0, 1)

mu = 0
sigma = 1

x2 = stats.norm(mu, sigma)

y = model(x1.rvs(samples), x2.rvs(samples))
kde = stats.gaussian_kde(y)
# %%
x = np.linspace(-10, 10, 100)

plt.plot(x, kde.pdf(x))
plt.hist(y, density=True)


# %%
x_test = np.linspace(-10, 10, 200)

plt.plot(x_test, kde_cdf(kde, x_test))
plt.step([-10, *exp_ou, 10], [0, *y_ecdf, 1], where="post")
plt.xlim([-10, 10])
plt.plot(exp_ou, kde_cdf(kde, exp_ou), "rd")

ds = kde_cdf(kde, exp_ou) - y_ecdf
ks = np.abs(ds).max()
print(f"{ks=}")


#%%
from scipy.optimize import minimize


def obj_ks(x0):
    samples = 200_000
    mu = x0[0]
    sigma = x0[1]
    x2 = stats.norm(mu, sigma)
    y = model(x1.rvs(samples, random_state=1), x2.rvs(samples, random_state=1))
    kde = stats.gaussian_kde(y)
    ds = kde_cdf(kde, exp_ou) - y_ecdf
    # print(f"{ks=}")
    return np.abs(ds).max()


# %%
x0 = np.array([0, 1])
res = minimize(obj_ks, x0, method="nelder-mead")
# res = minimize(obj_ks, x0)
res

# %%
y = model(
    x1.rvs(samples, random_state=1), stats.norm(*res.x).rvs(samples, random_state=1)
)
kde = stats.gaussian_kde(y)
# %%
x_test = np.linspace(-10, 10, 200)

plt.plot(x_test, kde_cdf(kde, x_test))
plt.step([-10, *exp_ou, 10], [0, *y_ecdf, 1], where="post")
plt.xlim([-10, 10])
plt.plot(exp_ou, kde_cdf(kde, exp_ou), "rd")
ds = kde_cdf(kde, exp_ou) - y_ecdf
ks = np.abs(ds).max()
print(f"{ks=}")
print(f"{res.x=}")

# %%
