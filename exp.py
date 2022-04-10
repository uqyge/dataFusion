import collections
import numpy as np
import matplotlib.pyplot as plt


def ecdf(exp):
    exp_ou = sorted(set(exp))
    freq = [collections.Counter(exp)[i] for i in exp_ou]
    # print(f"freq=")
    cdf = np.cumsum(freq) / sum(freq)
    # print(f"cdf=")
    return dict(zip(exp_ou, cdf))


if __name__ == "__main__":
    # %%
    exp = [-3, 1, 5]

    # %%
    exp_ou = sorted(set(exp))
    y = [ecdf(exp)[i] for i in exp_ou]

    plt.step([-10, *exp_ou, 10], [0, *y, 1], where="post")

    # %%
