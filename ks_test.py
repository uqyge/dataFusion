#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# %%
rng = np.random.default_rng()
# %%
samples = stats.uniform(-10, 20).rvs(100_000, random_state=rng)
# print(f"{samples=}")
#%%
res = stats.relfreq(samples, numbins=30)
x = res.lowerlimit + np.linspace(
    0, res.binsize * res.frequency.size, res.frequency.size
)

#%%
plt.plot(x, np.cumsum(res.frequency))

#%%
plt.plot(x, res.frequency, "kd")
# %%
stats.kstest(x, "norm")
# %%
np.cumsum(x)
# %%
res = stats.relfreq(x)
# %%
plt.plot(x, res)
# %%
res
# %%
res.frequency.shape
# %%
res
# %%
