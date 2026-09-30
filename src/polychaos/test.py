import numpy as np


def ishigami(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    ishi = np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)
    return ishi.reshape(-1, 1)


def branin(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (
        (x2 - 5.1 * x1 ** 2 / (4 * np.pi ** 2) + 5 * x1 / np.pi - 6) ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
