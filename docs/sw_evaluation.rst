.. _sw_evaluation:

Evaluation
==========
This module is for paralellizing likelihood estimation using the model.

When estimating parameters we differentiate between two layers of paralallization.

We can make seperate estimations for each subject or we can pool multiple subjects into one estimation. If the former is the case, the num_processes_subjs argument  in this module must be 1 (because the data set that is passed to it only includes the one subject). If, on the other hand we are pooling subjects the argument can be as large as the number of subjects in the pool.

num_processes_trials is the number of processes within each subject, i.e. the number of cores the trials are spread onto.

The main relevant function to call from the outside is

.. autofunction:: scenewalk.evaluation.evaluate_sw_parallel.get_neg_tot_like_parallel

Full Estimation Doc
===================
.. automodule:: scenewalk.evaluation.evaluate_sw_parallel
   :members:

