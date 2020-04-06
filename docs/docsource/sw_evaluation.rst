.. _sw_evaluation:

Evaluation
==========
This module is for paralellizing likelihood estimation using the model. When getting the model likelihood for large data sets or when computing it many times it may be beneficial to paralellize the get_scanpath_likelihood function.
This paralellized evaluateion of the model can happen at the level of subjects (run each on different cores) and/or at the level of trials (split trials into groups and run these groups in parallel).

´num_processes_subj´ is the number of processes between which the subjects are split.
´num_processes_trials´ is the number of processes within each subject, i.e. the number of cores the trials are spread onto.


The main relevant function to call from the outside is

.. autofunction:: scenewalk.evaluation.evaluate_sw_parallel.get_neg_tot_like_parallel

Full Estimation Doc
===================
.. automodule:: scenewalk.evaluation.evaluate_sw_parallel
   :members:

