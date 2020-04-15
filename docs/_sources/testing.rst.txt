.. _testing:

================
Testing
================

The Project code can (and should) be tested using the PyTest test suite in the tests folder.

Requirements:

- pytest
- pytest-cov (if you want to generate a coverage report)

Install the package with pip as described and then run the tests form the root directory of the project.

The command to run the core tests (which are halfway quick to run)::

	python3 -m pytest -m "basictest"

For all core tests::

	python3 -m pytest

In order to generate the coverage report run::

	python3 -m pytest -m "basictest"  --cov-report html --cov="."

