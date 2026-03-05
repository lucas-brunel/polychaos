from polychaos import PolyChaosExpansion

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    params = (
        [2, 4, 6],         # dim
        [3, 4, 5],         # deg
        ["uniform", "gaussian"]  # distribution
    )
    param_names = ['dim', 'deg', 'dist']

    def setup(self, dim, deg, dist):
        support = [[-1.0] * dim, [1.0] * dim]
        moments = [[0.0] * dim, [1.0] * dim]
        distributions = [dist] * dim

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

    def time_multi_index(self, dim, deg, dist):
        """Benchmark multi-index building."""
        self.pce._build_multi_index()

    def time_normalization_factors(self, dim, deg, dist):
        """Benchmark the normalization coefficients."""
        if not hasattr(self.pce, 'multi_index'):
             self.pce.multi_index = self.pce._build_multi_index()
        self.pce._normalization_factors()

#class MemSuite:
#    def mem_list(self):
#        return [0] * 256
