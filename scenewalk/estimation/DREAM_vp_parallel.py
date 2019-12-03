"""
Parallel Dream Interface for Scenewalk Model
Lisa Schwetlick 2019
University of Potsdam
"""

import multiprocessing as multiprocess
from multiprocessing.pool import ThreadPool
import numpy as np
from scenewalk.evaluation import evaluate_sw_parallel #get_total_neg_LL_per_subj(sw_model, x_dat_sub, y_dat_sub, dur_dat_sub, im_dat_sub, densities_dat)
from scenewalk.estimation import DREAM_param_estimation

def multi_process_like(args):
    """
    unpack args to multi process function
    """
    return DREAM_param_estimation.dream_estim_and_save(*args)


def do_multiprocess(function_args, num_processes):
    """
    Opens pool of workers for parallelization
    Inputs:
        - function_args: list of args to pass to Estimation
        - num_processes: how many pararell processes we want to run.
    """
    pool = ThreadPool(processes=num_processes)
    pool.map(multi_process_like, function_args)
    pool.close()
    pool.join()
    return True

def dream_and_save_parallel(sw_model, priors, default_params, x_dat, y_dat, dur_dat, im_dat, densities_dat, nchains, niter, vp_list, num_processes_dreams, num_processes_subjs, num_processes_trials):
    """
    Run multiple pydream jobs for individual subjects.
    Inputs:
        - sw_model: scenewalk model object
        - priors: ordered dict of priors for params to be estimated
        - default_params: ordered dict of priors (of not all parameters shoudl be estimated, default params provides the value for those params)
        - x_dat, y_dat, dur_dat, im_dat: data of shape subject[trial[path[]]]
        - densities_dat: empirical densities
        - nchains: number of PyDream chains to run
        - niter: number of PyDream iterations
        - vp_list: list of subject IDs to run
        - nprocesses: number of cores to parallelize the model evaluation over
    """
    assert num_processes_dreams == len(vp_list)
    assert num_processes_subjs == 1
    # prepare inputs for multi processing
    args_multi = []
    for vp in vp_list:
        # lik evaluation will run in a single process now
        args_multi.append((sw_model, priors, default_params, np.asarray([x_dat[vp]]), np.asarray([y_dat[vp]]), np.asarray([dur_dat[vp]]), np.asarray([im_dat[vp]]), densities_dat, num_processes_subjs, num_processes_trials, nchains, niter, vp))

    do_multiprocess(args_multi, num_processes_dreams)
