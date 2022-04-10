#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import kl_div


#%%
from exp import ecdf

# %%
exp = [13.43, 17.47, 14.22, 10.80, 10.24833, 13.79]
# exp = [i + 1 for i in exp]

#%%
exp_ou = sorted(set(exp))
y_ecdf = [ecdf(exp)[i] for i in exp_ou]
# delta = 0.2483
delta = 0.3
y_ecdf_u = [np.clip(ecdf(exp)[i] + delta, 0, 1) for i in exp_ou]
y_ecdf_l = [np.clip(ecdf(exp)[i] - delta, 0, 1) for i in exp_ou]

plt.step([0, *exp_ou, 30], [0, *y_ecdf, 1], where="post", label="cdf")
plt.step([0, *exp_ou, 30], [0, *y_ecdf_u, 1], where="post", label="upper bound")
plt.step([0, *exp_ou, 30], [0, *y_ecdf_l, 1], where="post", label="lower bound")
plt.legend()


#%%
from scipy.special import ndtr


def kde_cdf(kde, x):
    # return tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean() for item in x)
    return np.array(
        [ndtr(np.ravel(item - kde.dataset) / kde.factor).mean() for item in x]
    )
    # return ndtr(np.ravel(x - kde.dataset) / kde.factor).mean()


# %%
def model(p, t, r):
    # p, t, r = params
    E = 128e6
    nu = 0.24839

    # p = p * 6894.757
    p = max(0, p * 6894.757)
    t = t * 1e-6
    r = r * 1e-3

    sigma = ((-np.sqrt(2) / 9) * E * (t / r)) * np.sqrt(
        1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
    )
    out = p * np.pi * r**2 - 2 * np.pi * t * r * sigma
    return out * 0.5


# model(20, 64, 8.48)

# %%
def model2(params):
    p, t, r = params
    E = 128e6
    nu = 0.24839

    # p = p * 6894.757
    p = max(0, p * 6894.757)
    t = t * 1e-6
    r = r * 1e-3

    sigma = ((-np.sqrt(2) / 9) * E * (t / r)) * np.sqrt(
        1 / (1 - nu**2) + 4 * p / E * (r / t) ** 2
    )
    out = p * np.pi * r**2 - 2 * np.pi * t * r * sigma
    return out * 0.5


# model(20, 64, 8.48)

#%%
mu = 20
sigma = 1.5
p = stats.norm(mu, sigma)
t = stats.uniform(64, 66 - 64)
r = stats.uniform(8.48, 8.52 - 8.48)

# a, b, c = p.rvs(), t.rvs(), r.rvs()
# model(a, b, c)
model(20, 64, 8.48)
#%%
samples = 10_000
data = [p.rvs(samples), t.rvs(samples), r.rvs(samples)]
y = [model(data[0][i], data[1][i], data[2][i]) for i in range(samples)]
# y = model(p.rvs(samples), t.rvs(samples), r.rvs(samples))
kde = stats.gaussian_kde(y)
# %%
x = np.linspace(10, 25, 100)

plt.plot(x, kde.pdf(x))
plt.hist(y, bins=20, density=True)

#%%
%time
a = np.asarray(data).T
len(a)
#%%
%time
y = [model(*a[i]) for i in range(len(a))]

# print(f'{y.shape}')
#%%
%time
y = [model(data[0][i], data[1][i], data[2][i]) for i in range(samples)]

#%%
%time
y = [model2(a[i]) for i in range(len(a))]
# %%
x_test = np.linspace(0, 50, 200)

plt.plot(x_test, kde_cdf(kde, x_test))
plt.step([0, *exp_ou, 30], [0, *y_ecdf, 1], where="post", label="exp cdf")
plt.step([0, *exp_ou, 30], [0, *y_ecdf_u, 1], where="post", label="upper bound")
plt.step([0, *exp_ou, 30], [0, *y_ecdf_l, 1], where="post", label="lower bound")

plt.xlim([0, 50])
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
    p = stats.norm(mu, sigma)
    # y = model(
    #     p.rvs(samples, random_state=1),
    #     t.rvs(samples, random_state=1),
    #     r.rvs(samples, random_state=1),
    # )
    data = [
        p.rvs(samples, random_state=1),
        t.rvs(samples, random_state=1),
        r.rvs(samples, random_state=1),
    ]
    y = [model(data[0][i], data[1][i], data[2][i]) for i in range(samples)]

    kde = stats.gaussian_kde(y)
    ds = kde_cdf(kde, exp_ou) - y_ecdf
    # print(f"{ks=}")
    return np.abs(ds).max()


# %%
x0 = np.array([18, 1.5])
res = minimize(obj_ks, x0, method="nelder-mead")
# res = minimize(obj_ks, x0)
res

# %%
p = stats.norm(*res.x)
data = [
    p.rvs(samples, random_state=1),
    t.rvs(samples, random_state=1),
    r.rvs(samples, random_state=1),
]
y = [model(data[0][i], data[1][i], data[2][i]) for i in range(samples)]

kde = stats.gaussian_kde(y)
# %%
x_test = np.linspace(0, 30, 200)

plt.plot(x_test, kde_cdf(kde, x_test))
plt.step([0, *exp_ou, 30], [0, *y_ecdf, 1], where="post")
plt.step([0, *exp_ou, 30], [0, *y_ecdf_u, 1], where="post", label="upper bound")
plt.step([0, *exp_ou, 30], [0, *y_ecdf_l, 1], where="post", label="lower bound")

# plt.xlim([-10, 10])
plt.plot(exp_ou, kde_cdf(kde, exp_ou), "rd")
ds = kde_cdf(kde, exp_ou) - y_ecdf
ks = np.abs(ds).max()
print(f"{ks=}")
print(f"{res.x=}")

#%%
import jax
# %%
