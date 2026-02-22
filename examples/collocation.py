from src.polychaos import PolyChaosExpansion
import matplotlib.pyplot as plt
import numpy as np


def f(x): return x + np.sin(x)

print("\nV.A. uniformément distribuée")

support = [-1, 4]

xobs = np.random.random(30).reshape(-1, 1) * np.diff(support) + support[0]
x = np.linspace(*support, 201).reshape(-1, 1)
yobs = f(xobs)

pce = PolyChaosExpansion("uniform", support, 15)
pce.collocation(f, "gauss", 10)
y = pce.predict(x)

xmc = np.random.random(100_000)[:, None] * np.diff(support) + support[0]
ymc = pce.predict(xmc)

print(f"Mean: {pce.get_mean()} =? {np.mean(ymc)}")
print(f"Variance: {pce.get_var()} =? {np.var(ymc, ddof=1)}")

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(xobs, yobs, edgecolors="k", facecolors="none")
ax.plot(x, f(x), lw=1.2, c="k")
ax.plot(x, y, "-.b", lw=1.2)
ax.set_xlim(support)

print("\nV.A. normalement distribuée")

support = [-np.inf, np.inf]

xobs = np.random.random(100).reshape(-1, 1) * 12 - 6
x = np.linspace(-4, 4, 201).reshape(-1, 1)
yobs = f(xobs)

pce = PolyChaosExpansion("gaussian", support, 20, moments=[0.0, 1.0])
pce.collocation(f, "gauss", 20)
y = pce.predict(x)

xmc = np.random.randn(100_000)[:, None]
ymc = pce.predict(xmc)

print(f"Mean: {pce.get_mean()} =? {np.mean(ymc)}")
print(f"Variance: {pce.get_var()} =? {np.var(ymc, ddof=1)}")

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(xobs, yobs, edgecolors="k", facecolors="none")
ax.plot(x, f(x), lw=1.2, c="k")
ax.plot(x, y, "-.b", lw=1.2)

plt.show()
