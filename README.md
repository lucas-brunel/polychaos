# PolyChaos

## Installation

To use or develop this code, clone the repository and install it in "editable" mode. This ensures that any changes you make to the source code are instantly reflected when you import the package. To do this:

* Navigate to the root folder of the project
* Ensure you are using Python 3.10 or newer
* Run `pip install -e ".[test]"`

Note: The `[test]` flag automatically installs pytest and other development dependencies.

## Testing

To verify that the mathematics and logic are running correctly on your machine, simply run: 

```python
pytest
```

## Quickstart Guide

Here is a minimal example of training a 1D surrogate model using least-squares regression and extracting its exact analytical moments.

```python
import numpy as np
from polychaos import PolyChaosExpansion

# 1. Generate data (e.g., Uniform[-3, 3])
def expensive_model(x):
    return 1.0 + 2.0 * x - 0.5 * x**2 + 0.1 * x**3

x_train = np.random.uniform(-3, 3, size=(100, 1))
y_train = expensive_model(x_train)

# 2. Initialize and train the PCE model
pce = PolyChaosExpansion(distribution="uniform", support=[-3.0, 3.0], deg=5)
pce.regression(x_train, y_train)
# Supports also collocation

# 3. Make predictions on new data
x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
y_pred = pce.predict(x_test)

# 4. Extract moments computed analytically from the PCE coefficients
print(f"Analytical Mean: {pce.get_mean():.4f}")
print(f"Analytical Variance: {pce.get_var():.4f}")
```
