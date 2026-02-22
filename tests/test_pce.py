import pytest
import numpy as np
from numpy.testing import assert_allclose

from polychaos import PolyChaosExpansion

@pytest.fixture
def cubic_test():
    """Cubic polynomial."""
    return lambda x: 1.0 + 2.0 * x - 0.5 * x**2 + 0.1 * x**3

# Test PCE --------------------------------------------------------------------

@pytest.mark.parametrize("dist, support, moments", [
    ("uniform", [-3.0, 3.0], None),
    ("gaussian", [-np.inf, np.inf], [0.0, 1.0])
])
def test_regression_perfect_fit(cubic_test, dist, support, moments):
    """Test if PCE perfectly fits a polynomial of degree <= PCE degree."""

    if dist == "uniform":
        xobs = np.random.uniform(support[0], support[1], size=(100, 1))
    else:
        xobs = np.random.normal(moments[0], moments[1], size=(100, 1))

    yobs = cubic_test(xobs)

    # deg > 3 so should be perfect fit
    pce = PolyChaosExpansion(
        distribution=dist, support=support, deg=5, moments=moments
    )
    pce.regression(xobs, yobs)
    y_pred = pce.predict(xobs)

    assert_allclose(y_pred, yobs, atol=1e-10)

# Test custom errors ----------------------------------------------------------

def test_support_bounds_raises_error():
    """Test that reversed bounds trigger an error."""
    with pytest.raises(ValueError, match="Bounds should be strictly increasing."):
        PolyChaosExpansion("uniform", [1, -1], deg=5)

def test_invalid_degree_raises_error():
    """Test that invalid polynomial degree triggers an error."""
    with pytest.raises(ValueError, match="Invalid polynomial degree"):
        PolyChaosExpansion("uniform", [-1, 1], deg=0)
        
def test_invalid_distribution_raises_error():
    """Test that invalid distribution triggers an error."""
    with pytest.raises(ValueError, match="Invalid distribution"):
        PolyChaosExpansion("invalid", [-1, 1], deg=5)

def test_collocation_raises_error():
    """Test collocation errors."""
    pce = PolyChaosExpansion("gaussian", [-1, 1], deg=5)

    with pytest.raises(NotImplementedError):
        pce.collocation(cubic_test, "monte carlo", 10)

    with pytest.raises(ValueError, match="Invalid integration method"):
        pce.collocation(cubic_test, "invalid", 10)

    with pytest.raises(ValueError, match="Moments must be provided for Gaussian collocation."):
        pce.collocation(cubic_test, "gauss", 10)
