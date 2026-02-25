from itertools import product
from math import factorial
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.special import hermitenorm, legendre
from scipy.linalg import lstsq

class PolyChaosExpansion():
    """Polynomial Chaos Expansion [1]_.

    Parameters
    ----------
    distribution : str
        Input distribution, either `"uniform"`, `"gaussian"`
        (or `"normal"`).

    support : array_like(2, dim)
        Support of the input random variables.

    deg : int
        Truncation degree of the polynomial basis. Must be > 0.

    moments : array_like(2, dim), optional
        Default: `moments=None`. Mean and variance of the input if known,
        otherwise they are estimated from data. Required for the collocation
        estimation of the coefficients.

    truncation : str
        Truncation method. The only method availbe is `"total order"`.

    Attributes
    ----------
    coeffs : ndarray
        Computed PCE coefficients.

    References
    ----------
    .. [1] Xiu, D. (2010). Numerical Methods for Stochastic Computations: A
           Spectral Method Approach. Princeton University Press.
    """
    def __init__(
        self,
        distribution: str,
        support: npt.ArrayLike,
        deg: int,
        moments: npt.ArrayLike | None = None,
        truncation: str = "total order",
    ) -> None:
        support = np.array(support)
        if np.any(support[0] >= support[1]):
            raise ValueError(
                "Invalid bounds for support."
                " Bounds should be strictly increasing.")
        self.support = support  # Support of the input variables

        if deg < 1:
            raise ValueError(
                f"Invalid polynomial degree `{deg}`. Should be ≥ 1.")
        self.deg = deg  # Total order degree

        if truncation != "total order":
            raise NotImplementedError(
                "No other truncation scheme than 'total order'"
                " is available yet.")
        self.truncation = truncation  # Truncation method

        match distribution:
            case "uniform": p = legendre
            case "gaussian" | "normal": p = hermitenorm
            case _: raise ValueError(f"Invalid distribution '{distribution}'.")
        self.distribution = distribution

        self.dim = support.shape[1]  # Input dimensionality

        # Store the polynomials for later use, for efficiency
        self.polynomials = [p(i) for i in range(deg + 1)]

        # Input mean and variance if provided
        self.moments = np.array(moments) if moments is not None else None

        self.multi_index = self._build_multi_index()

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
                    2 * (self.xobs_ - self.support[[0]])
                    / np.diff(self.support, axis=0).reshape(1, -1)
                    - 1
                )
            case "gaussian" | "normal":
                if self.moments is None:
                    self.moments = np.array(
                        [np.mean(xobs, axis=0), np.var(xobs, axis=0, ddof=1)])
                self.xobs = (
                    (xobs - self.moments[[0]]) / np.sqrt(self.moments[[1]])
                )

        self.yobs = yobs.reshape(-1, 1)

        basis = self._build_basis(self.xobs)

        # Least-squares estimation of the PCE coefficients
        self.coeffs, _, _, _ = lstsq(basis.T, self.yobs)
        #                      ^^^^^^^^^^^^^^^^^^^^^^^^^
        # Better than bare np.linalg.inv(basis @ basis.T) @ basis @ self.yobs

    def collocation(self, f: Callable, method: str, n: int) -> None:
        """Compute the PCE coefficients with collocation.

        Parameters
        ----------
        f : function
            Function to be modeled. Should take an ndarray(1, dim) as input
            and return a float.

        method : str
            Integration scheme, either `"gauss"` (for Gauss-Legendre or
            Gauss-Hermite). `"monte carlo"` is planned but not yet available.

        n : int
            If `"gauss"`, `n` is the degree. If `"monte carlo"`, `n`
            is the number of points. Must be  > 0.
        """
        if method == "monte carlo":
            raise NotImplementedError()
        elif method != "gauss":
            raise ValueError(f"Invalid integration method '{method}'.")

        if not n > 0:
            raise ValueError(f"`n` should be > 0.")

        match self.distribution:
            case "uniform":
                sq_norms = np.array([
                    np.prod([1 / (2 * i + 1) for i in index])
                    for index in self.multi_index
                ])
                xint, wint = np.polynomial.legendre.leggauss(n)
                const = 0.5 ** self.dim
                normalize = lambda u: (
                    (0.5 * (u + 1))
                    * np.diff(self.support, axis=0).reshape(1, -1)
                    + self.support[[0]]
                )
            case "gaussian" | "normal":
                if self.moments is None:
                    raise ValueError(
                        "Moments must be provided for Gaussian collocation.")
                xint, wint = np.polynomial.hermite_e.hermegauss(n)
                sq_norms = np.array([
                    np.prod([factorial(i) for i in index if i != 1])
                    for index in self.multi_index
                ])
                const = (1.0 / np.sqrt(2.0 * np.pi)) ** self.dim
                normalize = lambda u: (
                    u * np.sqrt(self.moments[[1]]) + self.moments[[0]]
                )

        # Get the quadrature points
        xint_meshgrid = np.meshgrid(*[xint] * self.dim, indexing="ij")
        pts = np.stack(xint_meshgrid, axis=-1).reshape(-1, self.dim)
        #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Puts the dimension in x as the
        #              last so that it's ran through first when using reshape

        # Compute the quadrature weights as the product unidimensional weights
        wint_meshgrid = np.meshgrid(*[wint] * self.dim, indexing="ij")
        ws = np.prod(wint_meshgrid, axis=0)

        # Get the function value at the quadrature points in the right space
        ys = np.array([
            f(normalize(pt.reshape(1, -1))) for pt in pts
        ]).reshape(xint_meshgrid[0].shape)

        # Quadrature
        self.coeffs = np.array([
            const * np.sum(ws * ys * np.prod(
                [self.polynomials[i](xs) for xs, i in zip(xint_meshgrid, index)],
                axis=0
            )) / sn
            for index, sn in zip(self.multi_index, sq_norms)
        ]).reshape(-1, 1)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
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
                x = (
                    2 * (x - self.support[[0]])
                    / np.diff(self.support, axis=0).reshape(1, -1)
                    - 1
                )
            case "gaussian" | "normal":
                x = (x - self.moments[[0]]) / np.sqrt(self.moments[[1]])
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
        # Use the analytical expression of some integrals for efficiency
        # See https://dlmf.nist.gov/18.3
        match self.distribution:
            case "uniform":
                poly_sq_norms = np.array([
                    np.prod([1 / (2 * i + 1) for i in index])
                    for index in self.multi_index if np.sum(index) > 0
                ])
            case "gaussian" | "normal":
                poly_sq_norms = np.array([
                    np.prod([factorial(i) for i in index if i != 1])
                    for index in self.multi_index if np.sum(index) > 0
                ])

        return np.sum(poly_sq_norms * self.coeffs[1:].flatten() ** 2)

    def get_std(self) -> float:
        """Compute the standard deviation from the coefficients.

        Returns
        -------
        std : float
        """
        return np.sqrt(self.get_var())

    def _build_basis(self, x: npt.NDArray) -> npt.NDArray:
        """
        Parameters
        ----------
        x : ndarray(n, dim)

        Returns
        -------
        basis : ndarray(num_terms, n)
        """
        return np.array([
            np.prod([
                self.polynomials[i](x[:, d])
                for d, i in enumerate(index)
            ], axis=0)
            for index in self.multi_index
        ])

    def _build_multi_index(self) -> list[tuple]:
        """Generate multi-indices using total degree truncation.

        Returns
        -------
        orders : list of tuples
            List of multi-index, where each tuple has length `self.dim`.
        """
        # TODO: Super inefficient for high-dimensional cases
        prod_ = product(*[np.arange(self.deg + 1)] * self.dim)
        return [item for item in prod_ if np.sum(item) <= self.deg]
