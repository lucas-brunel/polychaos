from math import factorial
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.special import hermitenorm, legendre
from scipy.linalg import lstsq

class PolyChaosExpansion():
    """Polynomial Chaos Expansion.

    Parameters
    ----------
    distribution : str
        Input distribution, either ``"uniform"``, ``"gaussian"``
        (or ``"normal"``).

    support : array_like(2,)
        Support of the input random variable.

    deg : int
        Truncation degree of the polynomial basis. Must be > 0.

    moments : array_like(2,), optional
        Default: ``moments = None``. Mean and variance of the input if known,
        otherwise they are estimated from data. Required for the collocation
        estimation of the coefficients.

    Attributes
    ----------
    coeffs : ndarray
        Computed PCE coefficients.

    To Do
    -----
    * [x] Regression
    * [x] Prediction
    * [x] Scaling with support for uniform
    * [x] Scaling with mean and std for normal
    * [x] Get mean, variance, std with uniform
    * [x] Get mean, variance, std with gaussian
    * [x] Proper safety checks in `__init__`
    * [x] Distribution in `__init__` and other distributions
    * [x] Collocation
    * [ ] Multi-dimensional with the same distribution
    * [ ] Multi-dimensional with different distribution
    * [ ] As many distribution from the Askey scheme as possible
    * [ ] Truncation schemes:
        * [ ] Total-degree
        * [ ] Hyperbolic
        * [ ] Least Angle Regression
    """
    def __init__(
        self,
        distribution: str,
        support: npt.ArrayLike,
        deg :int,
        moments: npt.ArrayLike | None = None,
        #truncation="hyperbolic",
    ) -> None:
        if support[0] >= support[1]:
            raise ValueError(
                "Invalid bounds for support."
                " Bounds should be strictly increasing.")
        self.support = support

        if deg < 1:
            raise ValueError(
                f"Invalid polynomial degree `{deg}`. Should be ≥ 1.")
        self.deg = deg

        #if truncation is not "hyperbolic":
        #    raise NotImplementedError(
        #        "No other truncation scheme than 'hyperbolic'"
        #        " is available yet.")
        #self.truncation = truncation

        match distribution:
            case "uniform": p = legendre
            case "gaussian" | "normal": p = hermitenorm
            case _: raise ValueError(f"Invalid distribution '{distribution}'.")
        self.distribution = distribution

        self.polynomials = [p(i) for i in range(deg)]

        self.moments = moments

    def regression(self, xobs: npt.NDArray, yobs: npt.NDArray) -> None:
        """Compute the PCE coefficients with least-squares regression.

        Parameters
        ----------
        xobs : ndarray(nobs, dim)
            Input observations.

        yobs : ndarray(nobs, 1) or ndarray(nobs,)
            Output observations.
        """
        self.xobs_ = xobs
        match self.distribution:
            case "uniform":
                self.xobs = (
                    2 * (self.xobs_ - self.support[0]) / np.diff(self.support)
                    - 1
                )
            case "gaussian" | "normal":
                if not self.moments:
                    self.moments = [np.mean(xobs), np.var(xobs, ddof=1)]
                self.xobs = (xobs - self.moments[0]) / np.sqrt(self.moments[1])

        self.yobs = yobs.reshape(-1, 1)

        basis = self._build_basis(self.xobs)

        # Better than bare np.linalg.inv(basis @ basis.T) @ basis @ self.yobs
        self.coeffs, _, _, _ = lstsq(basis.T, self.yobs)

    def collocation(self, f: Callable, method: str, n: int) -> None:
        """Compute the PCE coefficients with collocation.

        Parameters
        ----------
        f : function
            Function to be modeled.

        method : str
            Integration scheme, either ``"gauss"`` (for Gauss-Legendre or
            Gauss-Hermite) or ``"monte carlo"``.

        n : int
            If ``"gauss"``, ``n`` is the degree. If ``"monte carlo"``, ``n``
            is the number of points. Must be  > 0.
        """
        if method == "monte carlo":
            raise NotImplementedError()
        elif method != "gauss":
            raise ValueError(f"Invalid integration method '{method}'.")

        match self.distribution:
            case "uniform":
                sq_norms = np.array([1 / (2 * i + 1) for i in range(self.deg)])
                xint, wint = np.polynomial.legendre.leggauss(n)
                const = 0.5
                xmapped = (
                    (0.5 * (xint + 1)) * np.diff(self.support)
                    + self.support[0]
                )
            case "gaussian" | "normal":
                if self.moments is None:
                    raise ValueError(
                        "Moments must be provided for Gaussian collocation.")
                xint, wint = np.polynomial.hermite_e.hermegauss(n)
                sq_norms = np.array([factorial(i) for i in range(self.deg)])
                const = 1.0 / np.sqrt(2.0 * np.pi)
                xmapped = xint * np.sqrt(self.moments[1]) + self.moments[0]

        yint = f(xmapped).flatten()

        self.coeffs = np.array([
            const * np.sum(wint * yint * p(xint)) / sn
            for p, sn in zip(self.polynomials, sq_norms)
        ]).reshape(-1, 1)

    def predict(self, x: npt.NDArray) -> np.ndarray:
        """Make predictions.

        Parameters
        ----------
        x : ndarray(n, dim)
            Prediction inputs.

        Returns
        -------
        y : ndarray(n, 1)
            Corresponding predictions.
        """
        match self.distribution:
            case "uniform":
                x = 2 * (x - self.support[0]) / np.diff(self.support) - 1
            case "gaussian" | "normal":
                x = (x - self.moments[0]) / np.sqrt(self.moments[1])
        basis = self._build_basis(x)
        return basis.T @ self.coeffs

    def get_mean(self) -> float:
        """Compute the mean from the coefficients.

        Returns
        -------
        mean : float
        """
        return self.coeffs[0].item()

    def get_var(self) -> float:
        """Compute the variance from the coefficients.

        Returns
        -------
        var : float
        """
        # Use the analytical norms of the polynomials for efficiency
        # See https://dlmf.nist.gov/18.3
        match self.distribution:
            case "uniform":
                poly_sq_norms = np.array(
                    [1 / (2 * i + 1) for i in range(1, self.deg)])
            case "gaussian" | "normal":
                poly_sq_norms = np.array(
                    [factorial(i) for i in range(1, self.deg)])

        return np.sum(self.coeffs[1:].flatten() ** 2 * poly_sq_norms)

    def get_std(self) -> float:
        """Compute the standard deviation from the coefficients.

        Returns
        -------
        std : float
        """
        return np.sqrt(self.get_var())

    def _build_basis(self, x: npt.NDArray) -> np.ndarray:
        """
        Parameters
        ----------
        x : ndarray(n, dim)

        Returns
        -------
        basis : ndarray(n, deg)
        """
        return np.array([p(x[:, 0]) for p in self.polynomials])
