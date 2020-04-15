.. _sw_estimation:

Estimation
==========
This module provides paralellization running multiple PyDream instances next to each other. Typically we use this for running one estimation for each subject to estimate separate parameters.
The main interface function of interest is dream_estim_and_save.

.. autofunction:: scenewalk.estimation.DREAM_param_estimation.dream_estim_and_save

Example Estimation Script
=========================
An example estimation project can be found under ``demo/estimation_pydream``.
The dream_estim.ipynb shows how to populate the arguments, priors and defaults.
The run_dream.py file shows how to call the estimation module.


Full Estimation Doc
===================
.. automodule:: scenewalk.estimation.DREAM_param_estimation
   :members:


.. automodule:: scenewalk.estimation.DREAM_vp_parallel
   :members:

