import numpy as np
from polychaos import PolyChaosExpansion


class TimePrivateMethods:
    params = (
        [2, 6, 10],  # dim
        [2, 6, 10],  # deg
        ["uniform", "gaussian"]  # distribution
    )
    param_names = ["dim", "deg", "dist"]

    def setup(self, dim, deg, dist):
        support = [[-1.0] * dim, [1.0] * dim]
        moments = [[0.0] * dim, [1.0] * dim]
        distributions = [dist] * dim

        self.dim = dim
        self.support = support

        np.random.seed(0)
        self.xtest = np.random.random((200, dim))

        try:
            # Try the modern API -> list of distributions
            self.pce = PolyChaosExpansion(distributions, support, deg, moments)
        except ValueError as e:
            if "Invalid distribution" in str(e):
                try:
                    # Fallback to the older API -> single string
                    self.pce = PolyChaosExpansion(dist, support, deg, moments)
                except Exception:
                    raise NotImplementedError("PCE setup not supported in this commit.")
            else:
                # If it's a different ValueError
                raise

        self.pce.multi_index = self.pce._build_multi_index()

    def time_multi_index(self, dim, deg, dist):
        """Benchmark multi-index building."""
        self.pce._build_multi_index()

    def time_build_basis(self, dim, deg, dist):
        """Benchmark basis building."""
        self.pce._build_basis(self.xtest)

    def time_normalization_factors(self, dim, deg, dist):
        """Benchmark the normalization coefficients."""
        self.pce._normalization_factors()


class TimeCollocationRegressionIshigami:
    params = (
        [10, 20],  # deg
        [10, 20],  # nint
    )
    param_names = ["deg", "nint"]

    def setup(self, deg, nint):
        dim = 3
        support = np.ones((2, dim))
        support[0] *= -np.pi
        support[1] *= np.pi
        moments = None
        distributions = ["uniform"] * dim

        self.support = support

        np.random.seed(0)
        self.xtest = np.random.random((200, dim))
        self.ytest = ishigami(self.xtest)

        try:
            # Try the modern API -> list of distributions
            self.pce = PolyChaosExpansion(distributions, support, deg, moments)
        except ValueError as e:
            if "Invalid distribution" in str(e):
                try:
                    # Fallback to the older API -> single string
                    self.pce = PolyChaosExpansion(
                        distributions[0], support, deg, moments)
                except Exception:
                    raise NotImplementedError("PCE setup not supported in this commit.")
            else:
                # If it's a different ValueError
                raise
        self.pce = PolyChaosExpansion(
            distributions, support, deg, moments)

        self.pce.collocation(ishigami, "gauss", nint)

    def time_collocation(self, deg, nint):
        """Benchmark collocation."""
        self.pce.collocation(ishigami, "gauss", nint)

    def mem_collocation(self, deg, nint):
        """Benchmark collocation."""
        self.pce.collocation(ishigami, "gauss", nint)

    def peakmem_collocation(self, deg, nint):
        """Benchmark collocation."""
        self.pce.collocation(ishigami, "gauss", nint)

    def time_regression(self, deg, nint):
        """Benchmark collocation."""
        self.pce.regression(self.xtest, self.ytest)

    def mem_regression(self, deg, nint):
        """Benchmark collocation."""
        self.pce.regression(self.xtest, self.ytest)

    def peakmem_regression(self, deg, nint):
        """Benchmark collocation."""
        self.pce.regression(self.xtest, self.ytest)

    def time_predict(self, deg, nint):
        """Benchmark prediction."""
        self.pce.predict(self.xtest)


def ishigami(x):
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    ishi = np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)
    return ishi.reshape(-1, 1)
