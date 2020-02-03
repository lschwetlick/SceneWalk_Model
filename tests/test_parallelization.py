"""
Tests for all paralellization
"""
import os
from collections import OrderedDict
import pytest
import numpy as np
from scenewalk.evaluation import  evaluate_sw_parallel
from scenewalk.scenewalk_model_object import scenewalk as sw_model
from scenewalk.utils import loadData

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

@pytest.mark.slowtest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_para_vs_seq_trials(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch):
    """
    Checks parallized result is the same as non parallelized evaluation
    """
    # Load Data
    dataDict = loadData.load_sim_data(os.path.join(THIS_DIR, 'test_simdata/'))
    x_dat, y_dat, dur_dat, im_dat, densities_dat, dat_range = loadData.dataDict2vars(dataDict)


    # x_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_x.npy'))
    # y_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_y.npy'))
    # dur_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_dur.npy'))
    # im_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_im.npy'))
    # densities_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/densities.npy'))
    # dat_range = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_range.npy'))

    #sw_params = list(meta[0].values())
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': dat_range[0], 'y': dat_range[1]})
    sw.update_params(sw_params)

    # get one subject dataset
    x_dat_sub = x_dat[0]
    y_dat_sub = y_dat[0]
    dur_dat_sub = dur_dat[0]
    im_dat_sub = im_dat[0]

    sequential_list = evaluate_sw_parallel.get_total_list_LL_trials(sw, x_dat_sub, y_dat_sub, dur_dat_sub, im_dat_sub, densities_dat)
    nLL_parallel = evaluate_sw_parallel.get_neg_tot_like_trials_parallel(sw, x_dat_sub, y_dat_sub, dur_dat_sub, im_dat_sub, densities_dat, 5)
    nLL_sequential = np.negative(np.sum(sequential_list))
    assert np.isclose(nLL_parallel, nLL_sequential)


@pytest.mark.slowtest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_para_vs_seq_both(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch):
    """
    Checks parallized result is the same as non parallelized evaluation
    """
    # Load Data
    dataDict = loadData.load_sim_data(os.path.join(THIS_DIR, 'test_simdata/'))
    x_dat, y_dat, dur_dat, im_dat, densities_dat, dat_range = loadData.dataDict2vars(dataDict)
    # x_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_x.npy'))
    # y_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_y.npy'))
    # dur_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_dur.npy'))
    # im_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_im.npy'))
    # densities_dat = np.load(os.path.join(THIS_DIR, 'test_simdata/densities.npy'))
    # dat_range = np.load(os.path.join(THIS_DIR, 'test_simdata/sim_range.npy'))

    #sw_params = list(meta[0].values())
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': dat_range[0], 'y': dat_range[1]})
    sw.update_params(sw_params)
    nLL_sequential = evaluate_sw_parallel.get_total_neg_LL_subjs(sw, x_dat, y_dat, dur_dat, im_dat, densities_dat, 1)
    nLL_parallel = evaluate_sw_parallel.get_neg_tot_like_parallel(sw, x_dat, y_dat, dur_dat, im_dat, densities_dat, 5, 3)
    assert np.isclose(nLL_parallel, nLL_sequential)
