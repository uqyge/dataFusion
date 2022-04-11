#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from hcsClass import ecdf, f

# %%
exp = [13.43, 17.47, 14.22, 10.80, 10.24833, 13.79]
exp_ou = sorted(set(exp))
y_ecdf = [ecdf(exp)[i] for i in exp_ou]

delta = 0.2483
# delta = 0.3
y_ecdf_u = [np.clip(ecdf(exp)[i] + delta, 0, 1) for i in exp_ou]
y_ecdf_l = [np.clip(ecdf(exp)[i] - delta, 0, 1) for i in exp_ou]

plt.step([0, *exp_ou, 40], [0, *y_ecdf, 1], where="post", label="cdf")
plt.step([0, *exp_ou, 40], [0, *y_ecdf_u, 1], "k--", where="post", label="upper bound")
plt.step([0, *exp_ou, 40], [0, *y_ecdf_l, 1], "k--", where="post", label="lower bound")
plt.fill_between(
    [0, *exp_ou, 40],
    [0, *y_ecdf_u, 1],
    [0, *y_ecdf_l, 1],
    step="post",
    alpha=0.5,
    color="grey",
)
plt.xlim([0, 30])
plt.legend()

#%%
def obj_KS(x0):
    mu = x0[0]
    sigma = x0[1]
    # mu, sigma = 20, 1.5
    c = x0[2]
    # c = 0.5
    ds = f([mu, sigma, c]).cdf(exp_ou) - y_ecdf
    # print(f"{ds=}")
    return np.abs(ds).max()


print(f"{obj_KS([20, 3, 0.5])=}")
# %%
x0 = np.array([18, 3, 0.5])
# x0 = np.array([20, 1.5])
res = minimize(
    obj_KS,
    x0,
    # method="COBYLA",
    method="nelder-mead",
    # method="SLSQP",
    # method="trust-constr",
)

res
# %%
x_test = np.linspace(0, 30, 100)
plt.plot(x_test, f(res.x).cdf(x_test), label="f*")
# plt.plot(x_test, f([13.8, 3, 0.5]).cdf(x_test), label="f*")
plt.plot(x_test, f(x0).cdf(x_test), label="f0")
# plt.plot(x_test, f([20, 1.5, 1]).cdf(x_test), label="f0")
plt.step([0, *exp_ou, 40], [0, *y_ecdf, 1], where="post")
plt.step([0, *exp_ou, 40], [0, *y_ecdf_u, 1], "k--", where="post", label="upper bound")
plt.step([0, *exp_ou, 40], [0, *y_ecdf_l, 1], "k--", where="post", label="lower bound")
plt.fill_between(
    [0, *exp_ou, 40],
    [0, *y_ecdf_u, 1],
    [0, *y_ecdf_l, 1],
    step="post",
    alpha=0.5,
    color="grey",
)
plt.xlim(
    [0, 30],
)
plt.legend()
plt.title("finding optimum cdf of Y based on maximum entropy")
# %%
plt.plot(x_test, f(res.x).pdf(x_test), label=f"{res.x.round(1)}")
plt.plot(x_test, f(x0).pdf(x_test), label=f"{x0}")
plt.legend()
plt.title("pdf of Y")

# %%
x_test = np.linspace(0, 40, 100)
plt.plot(x_test, stats.norm(*res.x[:2]).pdf(x_test), label=f"{res.x.round(1)}")
plt.plot(x_test, stats.norm(*x0[:2]).pdf(x_test), label=f"{x0}")
plt.legend()
plt.title("pdf of p")
# %%
x_test = np.linspace(0, 30, 100)
plt.plot(x_test, f(res.x).cdf(x_test), label="f*")
# plt.plot(x_test, f(x0).cdf(x_test), label="f0")
plt.plot(x_test, f([20, 1.5, 0.5]).cdf(x_test), label="f*")
# plt.plot(x_test, f([20, 1.5, 2]).cdf(x_test), label="f0")
plt.step([0, *exp_ou, 40], [0, *y_ecdf, 1], where="post")
plt.step([0, *exp_ou, 40], [0, *y_ecdf_u, 1], "k--", where="post", label="upper bound")
plt.step([0, *exp_ou, 40], [0, *y_ecdf_l, 1], "k--", where="post", label="lower bound")
plt.fill_between(
    [0, *exp_ou, 40],
    [0, *y_ecdf_u, 1],
    [0, *y_ecdf_l, 1],
    step="post",
    alpha=0.5,
    color="grey",
)
plt.xlim(
    [0, 30],
)
plt.legend()
plt.title("finding optimum cdf of Y based on maximum entropy")

# %%
