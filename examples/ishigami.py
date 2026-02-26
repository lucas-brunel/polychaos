from src.polychaos import PolyChaosExpansion

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

def ishigami(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    ishi = np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)
    return ishi.reshape(-1, 1)

support = np.ones((2, 3))
support[0] *= -np.pi
support[1] *= np.pi

Ns = [2 ** p for p in range(6, 20)]
print(f"Max number of samples: {Ns[-1]:,}")

xsample = np.random.random((Ns[-1], 3))
xsample *= support[[1]] - support[[0]]
xsample += support[[0]]
ysample = ishigami(xsample)

print(f"xsample: {xsample.nbytes / 1048576} MB")
print(f"ysample: {ysample.nbytes / 1048576} MB")

mean_mc, mean_pce, var_mc, var_pce = [], [], [], []

for N in Ns:
    xsubsample = xsample[:N]
    ysubsample = ysample[:N]
    
    mean_mc.append(np.mean(ysubsample))
    var_mc.append(np.var(ysubsample, ddof=1))

    pce = PolyChaosExpansion(
        distribution="uniform",
        support=support, 
        deg=10,
        moments=None,
        truncation="hyperbolic",
        q=0.75
    )
    pce.regression(xsubsample, ysubsample)

    mean_pce.append(pce.get_mean())
    var_pce.append(pce.get_var())

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.plot(Ns, mean_mc, "o-k", markersize=4, lw=1, label="MC mean")
ax1.plot(Ns, mean_pce, "s-b", markersize=4, lw=1, label="PCE mean")
ax1.set_xscale("log")
ax1.set_xlabel("Number of Monte Carlo samples")
ax1.set_ylabel("Mean")
ax1.legend(frameon=False)

ax2.plot(Ns, var_mc, "o-k", markersize=4, lw=1, label="MC variance")
ax2.plot(Ns, var_pce, "s-b", markersize=4, lw=1, label="PCE variance")
ax2.set_xscale("log")
ax2.set_xlabel("Number of Monte Carlo samples")
ax2.set_ylabel("Variance")
ax2.legend(frameon=False)

plt.show()
