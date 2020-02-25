import os
import sys
from collections import OrderedDict
import pytest
import numpy as np
from scenewalk.scenewalk_model_object import scenewalk as sw_model
from scenewalk.utils import loadData
import cProfile

sys.setrecursionlimit(10000)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#@profile
def bla():
    inhib_method = "subtractive"
    att_map_init_type = "cb"
    shifts = "both"
    exponents = 1
    locdep_decay_switch = "on"

    sw_params = OrderedDict({
        "omegaAttention" : 1,
        "omegaInhib": 0.1,
        "omfrac": 2,
        "sigmaAttention" : 5,
        "sigmaInhib" : 4,
        "gamma" : 1,
        "lamb" : 1,
        "inhibStrength" : 0.01,
        "zeta" : 0.01,
        "sigmaShift" : 5,
        "shift_size" : 2,
        "phi" : 100,
        "first_fix_OmegaAttention" : 3,
        "cb_sd_x" : 5,
        "cb_sd_y" : 4,
        "omega_prevloc": 1,
        "chi": 0.5,
        "ompfactor":0.1
    })

    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)}, {"omp":"add"})
    sw.update_params(sw_params)

    dataDict = loadData.load_sim_data('test_simdata/')
    x_sim, y_sim, dur_sim, _, densities, _ = loadData.dataDict2vars(dataDict)
    fix_dens = densities[0]

    avg_log_ll = sw.get_scanpath_likelihood(x_sim[0][0], y_sim[0][0], dur_sim[0][0], fix_dens)

if __name__ == '__main__':
    cProfile.run("bla()")