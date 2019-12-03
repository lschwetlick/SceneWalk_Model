"""
Dream Interface for Scenewalk Model
Lisa Schwetlick 2019
University of Potsdam
"""
import os
import sys
from argparse import ArgumentParser
import warnings
import pickle
from collections import OrderedDict
import time
import numpy as np
from scipy.stats import truncnorm as truncated_normal
from scipy.stats import uniform
#sys.path.append("..")
from scenewalk.pydream.core import run_dream as pd_run
from scenewalk.pydream.parameters import SampledParam as pd_param
from scenewalk.evaluation import evaluate_sw_parallel as evaluate_sw_parallel
from scenewalk.utils import utils
#from scenewalk.scenewalk_model_object import scenewalk


def trpd(my_mean, my_std, lb, ub):
    """
    Truncated normal prior distribution bounded between lb and ub with a sd of (ub-lb)/2 and a mean at the centre of the range
    Inputs:
        - my_mean: mean value of the truncated gaussian
        - my_std: standard deviation of the truncated gaussian
        - lb, ub: lower and upper bound of the truncated gaussian
    Returns:
        - pydream prior
    """
    a, b = (lb - my_mean) / my_std, (ub - my_mean) / my_std
    return pd_param(truncated_normal, a=a, b=b, scale=my_std, loc=my_mean)

def trunc_unif(lb, ub):
    """
    make uniform prior
    Inputs:
        - lb, ub: lower and upper bound of the truncated gaussian
    Returns:
        - pydream prior
    """
    loc = lb
    scale = ub-lb
    return pd_param(uniform, loc=loc, scale=scale)

def generate_custom_likelihood_function(sw_model, params, default_params, x_dat, y_dat, dur_dat, im_dat, densities_dat, num_processes_subjs, num_processes_trials):
    """
    generates a custom log likelihood function with the given setup
    ... wrapper, because pydream log likelihood function only works with one argument (the parvals)
    Inputs:
        - sw_model: scenewalk model object
        - params:
        - default_params:
        - x_dat, y_dat, dur_dat, im_dat: data of shape subject[trial[path[]]]
        - densities_dat: empirical densities
        - num_processes: number of cores to parallelize the model evaluation over
    Returns:
        - log likelihood fuction for the data
    """
    def custom_loglik(parvals):
        """
        evaluate scenewalk
        """
        sw_params = []
        fitted_params_names = params.keys()
        i = 0
        # enable fitting of only a few params
        for key in default_params.keys():
            if key in fitted_params_names:
                sw_params.append(parvals[i])
                i += 1
            else:
                sw_params.append(default_params[key])
        leftovers = sw_model.update_params(sw_params)
        assert leftovers is None, "giving too many parameters"
        sw_model.check_params_for_config()
        if sw_model.check_params_in_bounds():
            try:
                neg_like = evaluate_sw_parallel.get_neg_tot_like_parallel(sw_model, x_dat, y_dat, dur_dat, im_dat, densities_dat, num_processes_subjs, num_processes_trials=num_processes_trials)
            except:
                print("Model Failed with Parameters: ",sw_model.get_params())
                raise
            return - neg_like # well shit, turns out that neg neg is pos
        else:
            msg = str(["Params ran out of Bounds", parvals])
            warnings.warn(msg)
            return -np.inf
    return custom_loglik

def dream_estim_and_save(sw_model, priors, default_params, x_dat, y_dat, dur_dat, im_dat, densities_dat, num_processes_subjs, num_processes_trials, nchains, niter, vp_nr=None):
    """
    Run and Save dream chains
    Inputs:
        - sw_model: scenewalk model object
        - priors: ordered dict of priors for params to be estimated
        - default_params: ordered dict of priors (of not all parameters shoudl be estimated, default params provides the value for those params)
        - x_dat, y_dat, dur_dat, im_dat: data of shape subject[trial[path[]]]
        - densities_dat: empirical densities
        - num_processes: number of cores to parallelize the model evaluation over
        - nchains: number of PyDream chains to run
        - niter: number of PyDream iterations
        - vp_nr: id number of vp, for filename when running multiple estims
    Saves PyDream result
    """
    if vp_nr is not None:
        estim_id = (time.strftime("%Y%m%d-%H%M%S")) +"vp" + str(vp_nr)
    else:
        estim_id = (time.strftime("%Y%m%d-%H%M%S"))

    lik_func = generate_custom_likelihood_function(sw_model, priors, default_params, x_dat, y_dat, dur_dat, im_dat, densities_dat, num_processes_subjs, num_processes_trials)


    cwd = os.getcwd()
    os.mkdir(cwd + "/estim_%s" % estim_id)
    folderPath = cwd + "/estim_%s" % estim_id

    sampled_params, log_ps = pd_run(list(priors.values()), lik_func, nchains=nchains, niterations=niter, restart=False, verbose=False, model_name=folderPath+"fit_"+estim_id)
    # save chains
    np.save(folderPath + '/%s_estim_chains' % estim_id, sampled_params)
    np.save(folderPath + '/%s_estim_log_ps' % estim_id, log_ps)

    meta_dict = {
        "model_type": sw_model.whoami(),
        "len": len(dur_dat),
        "priors": priors,
        "defaults": default_params,
        "nchains": nchains,
        "niter": niter,
        "gitsha": utils.get_git_sha()
    }
    np.save(folderPath + '/%s_estim_meta' % estim_id, [meta_dict])

    # move dream files
    if vp_nr is None:
        os.rename("fit_" + estim_id + "_DREAM_chain_adapted_crossoverprob.npy", folderPath + "/fit_" + estim_id + "_DREAM_chain_adapted_crossoverprob.npy")
        os.rename("fit_" + estim_id + "_DREAM_chain_adapted_gammalevelprob.npy", folderPath + "/fit_" + estim_id + "_DREAM_chain_adapted_gammalevelprob.npy")
        os.rename("fit_" + estim_id + "_DREAM_chain_history.npy", folderPath + "/fit_" + estim_id + "_DREAM_chain_history.npy")



def main():
    """ Dream commandline interface"""
    parser = ArgumentParser()
    parser.add_argument('model',
                        metavar='m',
                        help='pickled scenewalk object')
    parser.add_argument('defaults',
                        metavar='d',
                        help='dict of default values as .npy')
    parser.add_argument('priors',
                        metavar='p',
                        help='dict of piors as .npy')
    parser.add_argument('nprocesses',
                        metavar='np',
                        help='number of parallel processes as int')
    parser.add_argument('nchains',
                        metavar='c',
                        help='number of chains to run')
    parser.add_argument('niter',
                        metavar='i',
                        help='number of iterations to run')
    parser.add_argument('data_path',
                        metavar='p',
                        help='path to where the data lies, up to')
    parser.add_argument('-densities',
                        metavar='dens',
                        required=False,
                        help='path to where the densitydata lies, up to')

    args = parser.parse_args()
    print("so far so good")
    with open(args.model, 'rb') as inp:
        sw_model = pickle.load(inp)
    priors_vals = np.load(args.priors).item()
    default_params = np.load(args.defaults).item()
    x_dat = np.load(args.data_path + "_x.npy")
    y_dat = np.load(args.data_path + "_y.npy")
    dur_dat = np.load(args.data_path + "_dur.npy")
    im_dat = np.load(args.data_path + "_im.npy")
    if args.densities is None:
        densities_dat = np.load(args.data_path + "_densities.npy")
    else:
        densities_dat = np.load(args.densities)
    print("loaded everything!")
    print(sw_model.whoami())

    priors = OrderedDict()
    for p in priors_vals.keys():
        priors[p] = trpd(*priors_vals[p])

    nprocesses = int(args.nprocesses)
    nchains = int(args.nchains)
    niter = int(args.niter)
    #print(type(int(args.nchains)))

    dream_estim_and_save(sw_model, priors, default_params, x_dat, y_dat, dur_dat, im_dat, densities_dat, nprocesses, nchains, niter)

if __name__ == "__main__":
    print("main")
    main()
