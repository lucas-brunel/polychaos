# PolyChaos

## Installation

To use or develop this code, clone the repository and install it in "editable" mode. This ensures that any changes you make to the source code are instantly reflected when you import the package. To do this:

* Navigate to the root folder of the project
* Ensure you are using Python 3.10 or newer
* Run `pip install -e ".[test]"`

Note: The `[test]` flag automatically installs pytest and other development and example dependencies.

## Examples

* [Surrogate model of the Branin function](examples/branin.py)
* [Convergence study of the mean and variance of the Ishigami function](examples/ishigami.py)

## Testing

To verify that the mathematics and logic are running correctly on your machine, simply run `pytest`.
