"""
Tests for all component functions of the scenewalk object
"""
import os
import sys
from collections import OrderedDict
import pytest
import numpy as np
from scenewalk.scenewalk_model_object import scenewalk as sw_model
from scenewalk.utils import loadData
from scenewalk.estimation import DREAM_param_estimation as dream_estim


default_params = OrderedDict({
    "omegaAttention" : None,
    "omegaInhib": None,
    "sigmaAttention" : None,
    "sigmaInhib" : 4,
    "gamma" : 1,
    "inhibStrength" : 0.01,
    "zeta" : None,
    "sigmaShift" : 5,
    "shift_size" : None,
    "first_fix_OmegaAttention" : 3,
    "cb_sd_x" : 5,
    "cb_sd_y" : 4,
    "omega_prevloc": None,
    "chi" : 3,
    "ompfactor":None,
})


priors = OrderedDict({
    "omegaAttention" : None,
    "omegaInhib": None,
    "sigmaAttention" : None,
    "zeta" : None,
    "shift_size" : None,
    "omega_prevloc" : None,
    "ompfactor" : None,
})

def test_param_passing():
    """checks that the routine for passing parameters through dream keeps the correct order!"""
    parvals = [123] * 8
    print(parvals)
    sw_params = dream_estim.param_list_from_estim_and_default(priors, default_params, parvals)
    print(sw_params)
    kwargs_d = {"saclen_shift":True, "omp":"add"}
    sw_args = ['subtractive', 'cb','both', 1, 'on', {'x': (0, 128), 'y': (0, 128)}, kwargs_d]
    sw = sw_model(*sw_args)
    sw.update_params(sw_params)
    recieved_params = sw.get_params()
    print(recieved_params)
    for key in recieved_params:
        print(key)
        if key in priors.keys():
            assert recieved_params[key] == 123
        else:
            assert recieved_params[key] == default_params[key]