.. _benchmarking:

================
Benchmarking
================

The model is defined for limited range of parameter values. This is the case because computers will flush numbers to zero if they become too small to save in the selected floating point representation. The SceneWalk model by default has an internal floating point representation of 128 bits (as opposed to the 64 bit standard float). This can be changed.

It is impossible to give a catch-all bounds specification, because the exact bounds depend on the exact specification of the model. The bounds of the parameters are very interdependent.

The file ``benchmarking/benchmark_sw.py`` provides a means of testing model bounds. It requires a model specification and a dictionary of parameters and associated bounds to be passed to it and will check all permutations of high and low bounds.


Example usage::

	python3 benchmarking/benchmark_sw.py "benchmarking/sw_args.npy" "benchmarking/bounds_dict.npy" "benchmarking/faulty.npy"


It will save all parameter combinations that made the model instable to a file (here: ``faulty.npy``) for further inspection.