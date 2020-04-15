import os
import sys
from collections import OrderedDict
import pytest
import numpy as np
from scenewalk.utils import loadData
from scenewalk.scenewalk_model_object import scenewalk as sw_model
from scenewalk.scenewalk_limited import limited_sw as lsw_model

sys.setrecursionlimit(10000)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "omega_prevloc" : 1
})

#@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_something(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch):
    dataDict = loadData.load_sim_data('tests/test_simdata/')
    chop_dataDict = loadData.chop_scanpaths(0, 10, dataDict)
    x_sim, y_sim, dur_sim, _, densities, rng = loadData.dataDict2vars(chop_dataDict)
    rng2 = {'x': rng[0],
            'y': rng[1]}
    fix_dens = densities[0]

    sw = sw_model(inhib_method, att_map_init_type, shifts, locdep_decay_switch, "off", rng2, {"exponents" : exponents})
    sw.update_params(sw_params)
    lsw = lsw_model(inhib_method, att_map_init_type, shifts, locdep_decay_switch, "off", rng2,{"n_history":10, "exponents" : exponents})
    lsw.update_params(sw_params)

    sw_LL = sw.get_scanpath_likelihood(x_sim[0][0], y_sim[0][0], dur_sim[0][0], fix_dens)
    lsw_LL = lsw.get_scanpath_likelihood(x_sim[0][0], y_sim[0][0], dur_sim[0][0], fix_dens)
    assert sw_LL == lsw_LL

#@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
@pytest.mark.parametrize('n_hist', (1, 2, 4, 8))
def test_contributing_fixs(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, n_hist):

    dataDict = loadData.load_sim_data('tests/test_simdata/')
    x_sim, y_sim, dur_sim, _, densities, rng = loadData.dataDict2vars(dataDict)
    rng2 = {'x': rng[0],
            'y': rng[1]}
    fix_dens = densities[0]

    lsw = lsw_model(inhib_method, att_map_init_type, shifts, locdep_decay_switch, "off", rng2, {"n_history":n_hist, "exponents" : exponents})
    lsw.update_params(sw_params)

    _, hist_list = lsw.get_scanpath_likelihood(x_sim[0][0], y_sim[0][0], dur_sim[0][0], fix_dens, debug=True)

    n_normal = [i[0] for i in hist_list if i[0] == 'normal']
    n_chopped = [i[0] for i in hist_list if i[0] == 'chopped']
    assert len(n_normal) == n_hist
    assert len(n_chopped) == (len(x_sim[0][0]) - len(n_normal) - 1)
    triplets = [i[1] for i in hist_list if i[0] == 'chopped']
    n_triplets = [len(i) for i in triplets]
    assert (np.asarray(n_triplets) == n_hist).all