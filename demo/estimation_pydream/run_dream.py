"""
Script to run dream estimation on the server
OMP
"""
import shutil
import sys
from collections import OrderedDict
import numpy as np
from scenewalk.scenewalk_model_object import scenewalk as scenewalk_obj
from scenewalk.estimation import DREAM_vp_parallel
from scenewalk.estimation import DREAM_param_estimation as dream_estim
from scenewalk.utils import loadData

np.seterr(invalid='raise')

def main():
    datadict = loadData.load_data("corpus_training")
    datadict = loadData.chop_scanpaths(1, 10, datadict)
    x_dat, y_dat, dur_dat, im_dat, densities_dat, dat_range = loadData.dataDict2vars(datadict)
    dat_range_x = dat_range[0]
    dat_range_y = dat_range[1]

    prior_args = np.load('priors.npy', allow_pickle=True).item()
    def_dict = np.load('defaults.npy', allow_pickle=True).item()


    priors = OrderedDict()
    for p in prior_args.keys():
        priors[p] = dream_estim.trpd(*prior_args[p])

    sw_args = np.load('sw_args.npy', allow_pickle=True)
    sw_model = scenewalk_obj(*sw_args)

    vp = int(sys.argv[1])-1

    # Ressource Allocation
    num_processes_subperdream = 1
    num_processes_trials = 6
    nchains = 3
    niter = 10

    dream_estim.dream_estim_and_save(sw_model, priors, def_dict, np.asarray([x_dat[vp]]), np.asarray([y_dat[vp]]),
                                    np.asarray([dur_dat[vp]]), np.asarray([im_dat[vp]]), densities_dat,
                                    num_processes_subperdream, num_processes_trials, nchains, niter,
                                    vp_nr=vp, destin_dir=".")

if __name__ == "__main__":
    main()