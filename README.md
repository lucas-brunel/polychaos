# PolyChaos

## Installation

To use or develop this code, clone the repository and install it in “editable” mode. This ensures that any changes you make to the source code are instantly reflected when you import the package. To do this:

* Navigate to the root folder of the project
* Ensure you are using Python 3.12 or newer
* Run `pip install -e ".[test]"`

Note: The `[test]` flag automatically installs pytest and other development and example dependencies.

## Examples

* [Surrogate model of the Branin function](examples/branin.py)
* [Convergence study of the mean and variance of the Ishigami function](examples/ishigami.py)

## Testing

This package uses _pytest_: To verify that the mathematics and logic are running correctly on your machine, simply run `pytest` in the terminal.

## Benchmarking

_Airspeed Velocity_, or `asv`, is used to benchmark this package.
If you want to perform the benchmark on the current package version, simply run `asv run` in the terminal (with `--quick` to avoid repetitions).
To perform the benchmark on all version—for example, to get the performance history—you can use `asv run ALL`.

For viewing the results, run `asv publish` and `asv preview`; then, simply open the URL provided in the terminal.
The database is cleared with `asv rm`.

For further details, see the [documentation](https://asv.readthedocs.io/en/stable/index.html).
