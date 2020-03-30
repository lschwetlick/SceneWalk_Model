"""
Parallel Dream Interface for Scenewalk Model

Lisa Schwetlick 2019

University of Potsdam
"""

import multiprocessing as multiprocess
from multiprocessing.pool import ThreadPool
import numpy as np
from scenewalk.evaluation import evaluate_sw_parallel
from scenewalk.estimation import DREAM_param_estimation

def multi_process_like(args):
    """
    unpack args into multi-process function
    """
    return DREAM_param_estimation.dream_estim_and_save(*args)


def do_multiprocess(function_args, num_processes):
    """
    Opens pool of workers for parallelization

    Parameters
    ----------
    function_args : list
        list of args to pass to Estimation
    num_processes : int
        how many pararell processes we want to run.

    Returns
    -------
    bool
        True if sucess
    """
    pool = ThreadPool(processes=num_processes)
    pool.map(multi_process_like, function_args)
    pool.close()
    pool.join()
    return True

def dream_and_save_parallel(sw_model, priors, default_params, x_dat, y_dat,
                            dur_dat, im_dat, densities_dat, nchains, niter,
                            vp_list, num_processes_dreams, num_processes_subjs,
                            num_processes_trials):
    """
    Run multiple pydream jobs for individual subjects.

    Parameters
    ----------
    sw_model : scenewalk model object
        scenewalk model object
    priors : dict
        dictionary of priors of the to be estimated parameters
    default_params : dict
        dictionary where the keys are all the parameters the model needs and to
        be estimated ones are None
    x_dat, y_dat, dur_dat, im_dat : arrays
        data of shape subject[trial[path[]]]
    densities_dat : array
        empirical densities
    nchains : int
        number of PyDream chains to run
    niter : int
        number of PyDream iterations
    vp_list : list
        list of id numbers of subjects
    num_processes_dreams : int
        number of dreams to run
    num_processes_subjs : int
        (only use when not estimating subjs individually)
        number of processes per subject
    num_processes_trials : int
        number of parallel threads per subject
    """
    assert num_processes_dreams == len(vp_list)
    assert num_processes_subjs == 1
    # prepare inputs for multi processing
    args_multi = []
    for vp in vp_list:
        # lik evaluation will run in a single process now
        args_multi.append((sw_model, priors, default_params,
                           np.asarray([x_dat[vp]]), np.asarray([y_dat[vp]]),
                           np.asarray([dur_dat[vp]]), np.asarray([im_dat[vp]]),
                           densities_dat, num_processes_subjs,
                           num_processes_trials, nchains, niter, vp))

    do_multiprocess(args_multi, num_processes_dreams)
