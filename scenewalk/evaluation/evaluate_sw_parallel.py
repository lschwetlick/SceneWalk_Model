"""
Get scenewalk log likelihood of dataset using paralellization
Lisa Schwetlick 2019
University of Potsdam
"""
import multiprocessing as multiprocess
from multiprocessing.pool import ThreadPool
import numpy as np

### ----------------------------------------------------------------
## These Functions parallelize over one suject's *trials*
### ----------------------------------------------------------------
def get_total_list_LL_trials(sw_model, x_dat, y_dat, dur_dat, im_dat,
                             densities_dat):
    """
    Gives a list containing a LL for each scanpath

    Parameters
    ----------
    sw_model : scenewalk model object
        scenewalk model object
    x_dat, y_dat, dur_dat, im_dat : arrays
        data of shape subject[trial[path[]]] either of whole dataset or of a
        subset (trials running in one pool)
    densities_dat : array
        empirical densities

    Returns
    -------
    array
        list of LLs for evaluation on subdataset
    """
    tr_LL_list = []
    for tr in range(len(x_dat)):
        im = im_dat[tr][0]
        path_likelihood = \
            sw_model.get_scanpath_likelihood(x_dat[tr], y_dat[tr],
                                             dur_dat[tr], densities_dat[im - 1])
        tr_LL_list.append(path_likelihood)

    #total_neg_LL = np.sum(sub_LL_list)
    return tr_LL_list

def multi_process_like_trials(args):
    """
    unpack args to multi process function
    """
    return get_total_list_LL_trials(*args)


def do_multiprocess_trials(function_args, num_processes):
    """
    Opens pool of workers for parallelization

    Parameters
    ----------
    function_args : list
        list of args to pass to Estimation
    num_processes : int
        how many pararell processes we want to run for trials (within subj)

    Returns
    -------
    list
        results_list: total sum for evaluation on dataset
    """
    if num_processes > 1:
        pool = multiprocess.Pool(processes=num_processes)
        results_list = pool.map(multi_process_like_trials, function_args)
        pool.close()
        pool.join()
    else:
        results_list = [get_total_list_LL_trials(*some_args) for \
            some_args in function_args]

    total_LL = np.sum(results_list)
    total_neg_LL = np.negative(total_LL)
    return total_neg_LL


def get_neg_tot_like_trials_parallel(sw_model, x_dat, y_dat, dur_dat, im_dat,
                                     densities_dat, num_processes):
    """
    Scenewalk evaluation parallelized by subject. Call this function from the
    outside by giving it one person's trials.

    Parameters
    ----------
    sw_model : scenewalk model object
        scenewalk model object
    x_dat, y_dat, dur_dat, im_dat : arrays
        data of shape subject[trial[path[]]] either of whole dataset or of a
        subset (trials running in one pool)
    densities_dat : array
        empirical densities
    num_processes : int
        number of cores to parallelize the model trial evaluation over

    Returns
    -------
    list
        results_list: total sum for evaluation on dataset
    """

    # prepare inputs for multi processing
    args_multi = []
    trial_indices = np.arange(len(x_dat))
    # make batches of subjects
    inputs = [trial_indices[i::num_processes] for i in range(num_processes)]
    for input_indices in inputs:
        args_multi.append((sw_model, x_dat[input_indices], y_dat[input_indices],
                           dur_dat[input_indices], im_dat[input_indices],
                           densities_dat))
    results = do_multiprocess_trials(args_multi, num_processes)
    return results

### ----------------------------------------------------------------
### These Functions pertain to estimating multiple *subjects*
### ----------------------------------------------------------------


def get_total_neg_LL_per_subj(sw_model, x_dat_sub, y_dat_sub, dur_dat_sub,
                              im_dat_sub, densities_dat):
    """
    Iterate over all images one subject saw and returns the negative Log
    Likelihood given parameters. (Linear)

    Parameters
    ----------
    sw_model : scenewalk model object
        scenewalk model object
    x_dat, y_dat, dur_dat, im_dat : arrays
        data of shape subject[trial[path[]]] either of whole dataset or of a
        subset (trials running in one pool)
    densities_dat : array
        empirical densities

    Returns
    -------
    float
        negative sum of likelihoods of all trials in a subject
    """
    sub_tr_LL_list = []
    for tr in range(len(x_dat_sub)):
        im = im_dat_sub[tr][0]
        path_likelihood = sw_model.get_scanpath_likelihood(x_dat_sub[tr],
                                                           y_dat_sub[tr],
                                                           dur_dat_sub[tr],
                                                           densities_dat[im-1])
        sub_tr_LL_list.append(path_likelihood)

    total_LL = np.sum(sub_tr_LL_list)
    total_neg_LL = np.negative(total_LL)
    return total_neg_LL


# Likelihood Funtion
def get_total_neg_LL_subjs(sw_model, x_dat, y_dat, dur_dat, im_dat,
                           densities_dat, num_processes_trials):
    """
    Calls get_total_neg_LL_per_subj() for every subject and returns the sum

    Parameters
    ----------
    sw_model : scenewalk model object
        scenewalk model object
    x_dat, y_dat, dur_dat, im_dat : arrays
        data of shape subject[trial[path[]]] either of whole dataset or of a
        subset (trials running in one pool)
    densities_dat : array
        empirical densities

    Returns
    -------
    float
        total sum for evaluation on subdataset
    """
    sub_LL_list = []
    if num_processes_trials > 1:
        for sub in range(len(x_dat)):
            sub_LL_list.append(\
                get_neg_tot_like_trials_parallel(sw_model,
                                                 x_dat[sub],
                                                 y_dat[sub],
                                                 dur_dat[sub],
                                                 im_dat[sub],
                                                 densities_dat,
                                                 num_processes_trials))
    else:
        for sub in range(len(x_dat)):
            sub_LL_list.append(get_total_neg_LL_per_subj(sw_model, x_dat[sub],
                                                         y_dat[sub],
                                                         dur_dat[sub],
                                                         im_dat[sub],
                                                         densities_dat))
    total_neg_LL = np.sum(sub_LL_list)
    return total_neg_LL


def multi_process_like_subjs(args):
    """
    unpack args to multi process function
    """
    return get_total_neg_LL_subjs(*args)


def do_multiprocess_subjs(function_args, num_processes):
    """
    Opens pool of workers for parallelization

    Parameters
    ----------
    function_args : list
        list of args to pass to Estimation
    num_processes : int
        how many pararell processes we want to run for subjects (between subj)

    Returns
    -------
    list
        results_list: total sum for evaluation on dataset
    """
    if num_processes > 1:
        pool = ThreadPool(processes=num_processes)
        results_list = pool.map(multi_process_like_subjs, function_args)
        pool.close()
        pool.join()
    else:
        results_list = [get_total_neg_LL_subjs(*some_args) for \
            some_args in function_args]
    return sum(results_list)


def get_neg_tot_like_parallel(sw_model, x_dat, y_dat, dur_dat, im_dat,
                              densities_dat, num_processes_subjs,
                              num_processes_trials=1):
    """
    Scenewalk evaluation parallelized by subject. This is what you call from
    the outside giving it a list of people

    Parameters
    ----------
    sw_model : scenewalk model object
        scenewalk model object
    x_dat, y_dat, dur_dat, im_dat : arrays
        data of shape subject[trial[path[]]] either of whole dataset or of a
        subset (trials running in one pool)
    densities_dat : array
        empirical densities
    num_processes_subjs : int
        number of cores to parallelize the model evaluation over. Max: number of
        subjects in the given dataset
    num_processes_trials : int
        number of cores to parallelize the model evaluation over. Max: number of
        trials in the given dataset

    Returns
    -------
    float

    """
    # prepare inputs for multi processing
    args_multi = []
    subject_indices = np.arange(len(x_dat))
    inputs = [subject_indices[i::num_processes_subjs] \
        for i in range(num_processes_subjs)]
    for input_indices in inputs:
        args_multi.append((sw_model, x_dat[input_indices], y_dat[input_indices],
                           dur_dat[input_indices], im_dat[input_indices],
                           densities_dat, num_processes_trials))
    results = do_multiprocess_subjs(args_multi, num_processes_subjs)
    return results
