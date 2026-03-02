from itertools import combinations
from math import factorial
from typing import Callable, Collection

import numpy as np
import numpy.typing as npt
from scipy.special import hermitenorm, legendre
from scipy.linalg import lstsq

class PolyChaosExpansion():
    """Polynomial Chaos Expansion [1]_ ,[2]_.

    Parameters
    ----------
    distribution : str
        Input distribution, either `"uniform"`, `"gaussian"`
        (or `"normal"`).

    support : array_like of shape (2, dim)
        Support of the input random variables.

    deg : int
        Truncation degree of the polynomial basis. Must be > 0.

    moments : array_like of shape (2, dim), optional
        Default: `moments=None`. Mean and variance of the input if known,
        otherwise they are estimated from data. Required for the collocation
        estimation of the coefficients.

    truncation : str, optional
        Truncation method. The only method availabe is `"total order"`.

    q : float, optional
        Default `q=1.0` (equivalent to total order). Defines the power of the
        hyperbolic truncation and must lie within (0,1]. Is ignored if
        `truncation="total order"`.

    Attributes
    ----------
    coeffs : ndarray
        Computed PCE coefficients.

    Examples
    --------
    Let :math:`X` be a random variable, with distribution
    :math:`\\mathcal{N}(0,1)`. We seek to model the random variable
    :math:`f(X)=2+3X-0.3X^2` using Polynomial Chaos Expansion.

    >>> import numpy as np
    >>> f = lambda x: 2 + 3 * x - 0.3 * x ** 2
    >>>
    >>> pce = PolyChaosExpansion(
    >>>     distribution="gaussian",
    >>>     support=[[-np.inf], [np.inf]],
    >>>     deg=3,
    >>>     moments=[[0.0], [1.0]]
    >>> )
    >>> pce.collocation(f=f, method="gauss", n=3)
    >>> pce.get_mean()
    np.float64(1.7)
    >>> pce.get_var()
    np.float64(9.18)

    References
    ----------
    .. [1] Xiu, D. (2010). Numerical Methods for Stochastic Computations: A
           Spectral Method Approach. Princeton University Press.
    .. [2] Lüthen, N., Marelli, S., & Sudret, B. (2021). Sparse Polynomial
           Chaos Expansions: Literature Survey and Benchmark. SIAM/ASA Journal
           on Uncertainty Quantification, 9(2), 593–649.
    """
    def __init__(
        self,
        distribution: str,
        support: npt.ArrayLike,
        deg: int,
        moments: npt.ArrayLike | None = None,
        truncation: str = "total order",
        q: float = 1.0,
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

        self.truncation = truncation  # Truncation method
        match self.truncation:
            case "total order": self.q = 1.0
            case "hyperbolic":
                if 0.0 < q <= 1.0:
                    self.q = q
                else:
                    raise ValueError("q must be such that 0 < q ≤ 1.")
            case _: raise ValueError(
                f"Invalid truncation method '{self.truncation}'.")

        self.distribution = distribution
        match self.distribution:
            case "uniform": p = legendre
            case "gaussian" | "normal": p = hermitenorm
            case _: raise ValueError(
                f"Invalid distribution '{self.distribution}'.")

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
        xobs : ndarray of shape (nobs, dim)
            Input observations.

        yobs : ndarray of shape (nobs, 1) or (nobs,)
            Output observations.
        """
        if self.distribution in {"gaussian", "normal"} and self.moments is None:
            self.moments = np.array(
                [np.mean(xobs, axis=0), np.var(xobs, axis=0, ddof=1)])

        self.xobs_ = xobs
        self.xobs = self._normalize(self.xobs_)

        self.yobs = yobs.reshape(-1, 1)

        basis = self._build_basis(self.xobs)

        # Least-squares estimation of the PCE coefficients
        self.coeffs, _, _, _ = lstsq(basis.T, self.yobs)
        #                      ^^^^^^^^^^^^^^^^^^^^^^^^^
        # Better than bare np.linalg.inv(basis @ basis.T) @ basis @ self.yobs

    def collocation(self, f: Callable, method: str, nint: int) -> None:
        """Compute the PCE coefficients with collocation.

        Parameters
        ----------
        f : function
            Function to be modeled. Should take an ndarray of shape (1, dim)
            as input and return a float.

        method : str
            Integration scheme, either `"gauss"` (for Gauss-Legendre or
            Gauss-Hermite). `"smolyak"` is planned but not yet available.

        nint : int
            If `"gauss"`, `nint` is the number of point per dimension.
        """
        if method == "smolyak":
            raise NotImplementedError()
        elif method != "gauss":
            raise ValueError(f"Invalid integration method '{method}'.")

        if not nint > 0:
            raise ValueError(f"`n` should be > 0.")

        # Quadrature points, weights, and constant
        xint_meshgrid, ws, const = self._quadrature(nint)

        gamma = self._normalization_factors()

        # Get the function value at the quadrature points in the right space
        pts = np.stack(xint_meshgrid, axis=-1).reshape(-1, self.dim)
        #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Puts the dimension in x as the
        #              last so that it's ran through first when using reshape
        ys = np.array([
            f(self._denormalize(pt.reshape(1, -1))) for pt in pts
        ]).reshape(xint_meshgrid[0].shape)

        # Quadrature
        self.coeffs = np.array([
            const * np.sum(ws * ys * np.prod(
                [self.polynomials[i](xs) for xs, i in zip(xint_meshgrid, index)],
                axis=0
            )) / sn
            for index, sn in zip(self.multi_index, gamma)
        ]).reshape(-1, 1)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """Make predictions.

        Parameters
        ----------
        x : ndarray of shape (n, dim)
            Prediction inputs.

        Returns
        -------
        y : ndarray of shape (n, 1)
            Corresponding predictions.
        """
        x = self._normalize(x)
        basis = self._build_basis(x)
        return basis.T @ self.coeffs

    def get_mean(self) -> float:
        """Compute the mean from the coefficients.

        Returns
        -------
        mean : float
        """
        return self.coeffs[0][0]

    def get_var(self) -> float:
        """Compute the variance from the coefficients.

        Returns
        -------
        var : float
        """
        gamma = self._normalization_factors()[1:]
        return np.sum(gamma * self.coeffs[1:, 0] ** 2)

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
        x : ndarray of shape (n, dim)

        Returns
        -------
        basis : ndarray of shape (num_terms, n)
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

        Notes
        -----
        See https://math.stackexchange.com/questions/36250/number-of-monomials-of-certain-degree
        """
        arr = list(combinations(range(self.deg + self.dim), self.dim))
        degrees = np.diff(arr, axis=1, prepend=-1) - 1
        mask = np.sum(degrees ** self.q, axis=1)  ** (1 / self.q) > self.deg
        return degrees[~mask, :]

    def _normalization_factors(self) -> npt.NDArray:
        """Generate the normalization factors γ.

        Returns
        -------
        gamma : ndarray
            The normalization factors corresponding to all the elements of the
            multi-index.
        """
        # Use the analytical expression of some integrals for efficiency
        # See https://dlmf.nist.gov/18.3
        match self.distribution:
            case "uniform":
                gamma = np.array([
                    np.prod([1 / (2 * i + 1) for i in index])
                    for index in self.multi_index
                ])
            case "gaussian" | "normal":
                gamma = np.array([
                    np.prod([factorial(i) for i in index])
                    for index in self.multi_index
                ])
        return gamma

    def _normalize(self, x: npt.NDArray) -> npt.NDArray:
        """Normalize data depending on the distribution.
        
        Parameters
        ----------
        x : ndarray (nx, dim)

        Returns
        -------
        x_normalized : ndarray (nx, dim) 
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

        return x

    def _denormalize(self, x: npt.NDArray) -> npt.NDArray:
        """Denormalize data depending on the distribution.
        
        Parameters
        ----------
        x : ndarray (nx, dim)

        Returns
        -------
        x_denormalized : ndarray (nx, dim) 
        """
        match self.distribution:
            case "uniform":
                x = (
                    (0.5 * (x + 1))
                    * np.diff(self.support, axis=0).reshape(1, -1)
                    + self.support[[0]]
                )
            case "gaussian" | "normal":
                x = x * np.sqrt(self.moments[[1]]) + self.moments[[0]]

        return x

    def _quadrature(self, nint: int) -> tuple:
        """Generates the quadrature points, their weights, and the integration
        constant.

        Constructs a full tensor-product grid for numerical integration based
        on the underlying probability measure (Legendre-Gauss for uniform,
        probabilist's Hermite-Gauss for normal).

        Parameters
        ----------
        nint : int
            Number of points per dimension for the quadrature.

        Returns
        -------
        xint_meshgrid : list of dim ndarrays of size (nint,) * dim
            Quadrature points.

        weights : ndarray of size (nint,) * dim
            Quadrature weights.

        const : float
            Quadrature constant.
        """
        match self.distribution:
            case "uniform":
                xint, wint = np.polynomial.legendre.leggauss(nint)
                const = 0.5 ** self.dim
            case "gaussian" | "normal":
                if self.moments is None:
                    raise ValueError(
                        "Moments must be provided for Gauss collocation.")
                xint, wint = np.polynomial.hermite_e.hermegauss(nint)
                const = (1.0 / np.sqrt(2.0 * np.pi)) ** self.dim

        # Get the quadrature points
        xint_meshgrid = np.meshgrid(*[xint] * self.dim, indexing="ij")

        # Compute the quadrature weights as the product unidimensional weights
        wint_meshgrid = np.meshgrid(*[wint] * self.dim, indexing="ij")
        weights = np.prod(wint_meshgrid, axis=0)

        return (xint_meshgrid, weights, const)
