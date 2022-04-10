# %%
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt


#%%
x = np.linspace(-5, 5, 20)
# %%
def gauss(mu=0, sigma=1):
    return stats.norm(mu, sigma)


gauss1 = gauss()
# %%
plt.plot(x, gauss1.pdf(x))

plt.plot(x, gauss1.cdf(x))

#%%
plt.plot(x, gauss1.pdf(x))
plt.plot(x, gauss(-1, 1).pdf(x))

# %%
a = stats.norm(0, 1)
b = stats.norm(0, 1)
# %%
def test(sigma):
    return a.rvs() + sigma if b.rvs() > 1 / 3 else a.rvs() - sigma


#%%
samples = [test(2) for _ in range(10_000)]
# %%
plt.hist(samples, bins=40)
# %%
test(2)
# %%
b > 1 / 3
# %%
def f0(x):
    return 2*x
# %%
x = np.linspace(0,1,10)
# %%
plt.plot(x,f0(x))
# %%
x0=stats.norm(0, 1).rvs(100)
# %%
plt.plot(x0,'d')
# %%
# f0(x0)
plt.hist(f0(x0))

# %%
