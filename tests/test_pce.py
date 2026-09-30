import pytest
import numpy as np
from numpy.testing import assert_allclose

from polychaos import PolyChaosExpansion


@pytest.fixture
def nd_poly_test():
    """2D polynomial of total degree 3."""
    # f(x1, x2) = 1.0 + 2.0*x1 - 0.5*x2^2 + 0.1*x1*(x2^2)
    return lambda x: (
        1.0 + 2.0 * x[:, 0] - 0.5 * x[:, 1]**2 + 0.1 * x[:, 0] * x[:, 1]**2
    ).reshape(-1, 1)

# Test PCE --------------------------------------------------------------------

@pytest.mark.parametrize("dist, support, moments", [
    ("uniform", [[-3.0, -2.0], [3.0, 2.0]], None),
    ("gaussian", [[-np.inf, -np.inf], [np.inf, np.inf]], [[0.0, 1.0], [1.0, 2.0]])
])
def test_regression_perfect_fit_nd(nd_poly_test, dist, support, moments):
    """Test if PCE perfectly fits a 2D polynomial of degree <= PCE degree."""
    
    dim = len(support[0])
    n_samples = 200

    if dist == "uniform":
        xobs = np.random.uniform(support[0], support[1], size=(n_samples, dim))
    else:
        xobs = np.random.normal(moments[0], np.sqrt(moments[1]), size=(n_samples, dim))

    yobs = nd_poly_test(xobs)

    # Polynomial total degree = 3 => `deg=4`` should be a perfect fit
    pce = PolyChaosExpansion(
        distribution=[dist] * dim, support=support, deg=4, moments=moments
    )
    pce.regression(xobs, yobs)
    y_pred = pce.predict(xobs)

    assert_allclose(y_pred, yobs, atol=1e-10)


@pytest.mark.parametrize("dist, support, moments", [
    ("uniform", [[-3.0, -2.0], [3.0, 2.0]], None),
    ("gaussian", [[-np.inf, -np.inf], [np.inf, np.inf]], [[0.0, 1.0], [1.0, 2.0]])
])
def test_collocation_perfect_fit_nd(nd_poly_test, dist, support, moments):
    """Test if PCE perfectly fits a 2D polynomial using collocation."""

    dim = len(support[0])
    n_samples = 100

    if dist == "uniform":
        xobs = np.random.uniform(support[0], support[1], size=(n_samples, dim))
    else:
        xobs = np.random.normal(moments[0], np.sqrt(moments[1]), size=(n_samples, dim))

    yobs = nd_poly_test(xobs)

    pce = PolyChaosExpansion(
        distribution=[dist] * dim, support=support, deg=4, moments=moments
    )
    
    # n=5 provides enough quadrature points per dimension for degree 4
    pce.collocation(nd_poly_test, "gauss", 5)
    y_pred = pce.predict(xobs)

    assert_allclose(y_pred, yobs, atol=1e-10)


def test_docstring_example():
    """Make sure the docstring example returns the analytical values."""
    f = lambda x: 2 + 3 * x - 0.3 * x ** 2

    pce = PolyChaosExpansion(
        distribution=["gaussian"],
        support=[[-np.inf], [np.inf]],
        deg=3,
        moments=[[0.0], [1.0]]
    )
    pce.collocation(f=f, method="gauss", nint=3)

    assert_allclose(pce.get_mean(), 1.7, atol=1e-10)
    assert_allclose(pce.get_var(), 9.18, atol=1e-10)
    assert_allclose(pce.get_std(), np.sqrt(9.18), atol=1e-10)


def test_mixed_distribution_pce():
    """
    Test PCE combining Uniform and Gaussian distributions against an 
    exact analytical polynomial benchmark.
    
    Model: Y = X_1 + X_2^2 + X_1 * X_2
    Inputs: X_1 ~ U(-1, 1), X_2 ~ N(0, 1)
    Analytical Mean: 1.0
    Analytical Variance: 8/3
    """
    f = lambda x: x[:, 0] + x[:, 1]**2 + x[:, 0] * x[:, 1]

    support = [
        [-1.0, -np.inf], 
        [ 1.0,  np.inf]
    ]
    
    moments = [
        [0.0, 0.0], 
        [0.0, 1.0]
    ]

    pce = PolyChaosExpansion(
        distribution=["uniform", "gaussian"],
        support=support,
        deg=2,
        moments=moments
    )
    
    pce.collocation(f=f, method="gauss", nint=3)

    assert_allclose(pce.get_mean(), 1.0, atol=1e-10)
    assert_allclose(pce.get_var(), 8.0 / 3.0, atol=1e-10)

# Test custom errors ----------------------------------------------------------

def test_support_bounds_raises_error():
    """Test that reversed bounds trigger an error."""
    with pytest.raises(ValueError, match="Bounds should be strictly increasing."):
        PolyChaosExpansion(["uniform"], [[1], [-1]], deg=5)


def test_invalid_degree_raises_error():
    """Test that invalid polynomial degree triggers an error."""
    with pytest.raises(ValueError, match="Invalid polynomial degree"):
        PolyChaosExpansion(["uniform"], [[-1], [1]], deg=0)

    
def test_invalid_distribution_raises_error():
    """Test that invalid distribution triggers an error."""
    with pytest.raises(ValueError, match="Invalid distribution"):
        PolyChaosExpansion(["invalid"], [[-1], [1]], deg=5)


def test_invalid_distribution_dimension_raises_error():
    """Test that invalid distribution triggers an error."""
    with pytest.raises(ValueError, match="`distribution` should be a list of"):
        PolyChaosExpansion(["uniform"] * 3, [[-1], [1]], deg=5)


def test_invalid_truncation_raises_error():
    """Test that invalid truncation scheme triggers an error."""
    with pytest.raises(ValueError, match="Invalid truncation method"):
        PolyChaosExpansion(["uniform"], [[-1], [1]], deg=5, truncation="invalid")


def test_invalid_q_raises_error():
    """Test that invalid q triggers an error."""
    with pytest.raises(ValueError, match="q must be such that 0 < q ≤ 1."):
        PolyChaosExpansion(
            ["uniform"], [[-1], [1]], deg=5, truncation="hyperbolic", q=0.0)


def test_collocation_raises_error():
    """Test collocation errors."""
    pce = PolyChaosExpansion(["gaussian"], [[-1], [1]], deg=5)

    with pytest.raises(NotImplementedError):
        pce.collocation(nd_poly_test, "smolyak", 10)

    with pytest.raises(ValueError, match="`n` should be > 0."):
        pce.collocation(nd_poly_test, "gauss", 0)

    with pytest.raises(ValueError, match="Invalid integration method"):
        pce.collocation(nd_poly_test, "invalid", 10)

    with pytest.raises(ValueError, match="Moments must be provided for Gauss collocation."):
        pce.collocation(nd_poly_test, "gauss", 10)
