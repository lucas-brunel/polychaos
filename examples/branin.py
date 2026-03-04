"""Branin function.

Example extracted from https://www.sfu.ca/~ssurjano/branin.html
"""

from src.polychaos import PolyChaosExpansion

import matplotlib.pyplot as plt
import numpy as np

def branin(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (
        (x2 - 5.1 * x1 ** 2 / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
        + 10
    )

# Input domain and visualization grid

x1support = [-5, 10]
x2support = [0, 15]
support = np.array([x1support, x2support]).T
# ^^^^^ [[-5  0]
#        [10 15]]

x1 = np.linspace(*x1support)
x2 = np.linspace(*x2support)

X1, X2 = np.meshgrid(x1, x2, indexing="ij")

# True function

x = np.stack((X1, X2), axis=-1).reshape(-1, 2)
Y = np.array([branin(xi.reshape(1, -1)) for xi in x]).reshape(X1.shape)

# Training data

N = 100
xobs = np.random.random((N, 2))
xobs *= (support[[1]] - support[[0]])
xobs += support[[0]]
yobs = branin(xobs).reshape(-1, 1)

# PCE using regression to approximate the coefficients

pce_reg = PolyChaosExpansion(
    ["uniform"] * 2, support, 10, truncation="hyperbolic", q=0.5)
pce_reg.regression(xobs, yobs)
ypce_reg = pce_reg.predict(x)
Ypce_reg = ypce_reg.reshape(X1.shape)

# PCE using collocation to approximate the coefficients

pce_col = PolyChaosExpansion(
    ["uniform"] * 2, support, 10, truncation="hyperbolic", q=0.5)
pce_col.collocation(branin, "gauss", int(N ** 0.5))
ypce_col = pce_col.predict(x)
Ypce_col = ypce_col.reshape(X1.shape)

# Visualization

vmin = min(Y.min(), Ypce_reg.min(), Ypce_col.min())
vmax = max(Y.max(), Ypce_reg.max(), Ypce_col.max())
levels = np.linspace(vmin, vmax, 11)

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

axes[0].set_aspect("equal")
cs0 = axes[0].contourf(X1, X2, Y, levels=levels)
axes[0].set_title("True function")

axes[1].set_aspect("equal")
cs1 = axes[1].contourf(X1, X2, Ypce_reg, levels=levels)
axes[1].set_title("PCE with regression")

axes[2].set_aspect("equal")
cs1 = axes[2].contourf(X1, X2, Ypce_col, levels=levels)
axes[2].set_title("PCE with collocation")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(cs0, cax=cbar_ax)

plt.show()
