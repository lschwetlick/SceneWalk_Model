"""
tests for simulation module
"""
from collections import OrderedDict
import os
import shutil
import numpy as np
import pytest
from scenewalk.simulation import simulate_dataset as sim
from scenewalk.scenewalk_model_object import scenewalk as scenewalk_obj
from scenewalk.utils import loadData

@pytest.mark.basictest
def test_sample_from_chains_dict():
    vp_params = np.load("tests/test_estimation/sample_dict.npy", allow_pickle=True).item()
    testsub = np.random.randint(0, len(vp_params.keys()))
    param_dict = sim._sample_from_chains_dict(vp_params[testsub])
    assert all([isinstance(v, (int, float, np.int64)) for v in param_dict.values()])
    print()
    for p in param_dict:
        if not isinstance(vp_params[testsub][p], (list, np.ndarray)):
            assert param_dict[p] == vp_params[testsub][p]
        else:
            assert param_dict[p] in vp_params[testsub][p]

@pytest.mark.basictest
def test_simulate_by_subj():
    dataDict = loadData.load_sim_data('tests/test_simdata/')
    x, y, dur_dat, im_dat, densities_dat, d_range = loadData.dataDict2vars(dataDict)

    kwargs_d = {"coupled_oms": True, "exponents":1}
    sw_args = ['subtractive', 'zero', 'off', "off", 'off', {'x': d_range[0], 'y': d_range[1]}, kwargs_d]
    sw_model = scenewalk_obj(*sw_args)
    params_by_sub = np.load("tests/test_estimation/point_est_subj.npy", allow_pickle=True).item()

    sim_id = sim.simulate(dur_dat, im_dat, densities_dat, sw_model, params=params_by_sub, start_loc="data", x_path=x, y_path=y, resample_durs=True)
    #sim_id = "sim_20191025-101400"
    sim_id = sim_id[4:]
    cwd = os.getcwd()
    folderPath = cwd + "/sim_%s" % sim_id

    simdataDict = loadData.load_sim_data(folderPath+'/')
    x_sim, y_sim, dur_sim, im_sim, _, _ = loadData.dataDict2vars(simdataDict)

    for s, o in zip([x_sim, y_sim, dur_sim, im_sim], [x, y, dur_dat, im_dat]):
        assert len(s) == len(o)
        assert type(s) == type(o)
        testsub = np.random.randint(len(o))
        assert len(s[testsub]) == len(o[testsub])
        assert type(s[testsub]) == type(o[testsub])
        testtrial = np.random.randint(len(o[testsub]))
        assert len(s[testsub][testtrial]) == len(o[testsub][testtrial])
        assert type(s[testsub][testtrial]) == type(o[testsub][testtrial])
    shutil.rmtree(folderPath)


@pytest.mark.basictest
def test_simulate_sample():
    dataDict = loadData.load_sim_data('tests/test_simdata/')
    x, y, dur_dat, im_dat, densities_dat, d_range = loadData.dataDict2vars(dataDict)

    kwargs_d = {"coupled_oms": True, "exponents":1}
    sw_args = ['subtractive', 'zero', 'off', "off", 'off', {'x': d_range[0], 'y': d_range[1]}, kwargs_d]
    sw_model = scenewalk_obj(*sw_args)
    chains_dict = np.load("tests/test_estimation/sample_dict.npy", allow_pickle=True).item()

    sim_id = sim.simulate_sample(dur_dat, im_dat, densities_dat, sw_model, chains_dict, "trial", start_loc="center", x_path=None, y_path=None, resample_durs=False)
    #sim_id = "sim_20191025-101400"
    sim_id = sim_id[4:]
    cwd = os.getcwd()
    folderPath = cwd + "/sim_%s" % sim_id

    simdataDict = loadData.load_sim_data(folderPath+'/')
    x_sim, y_sim, dur_sim, im_sim, _, _ = loadData.dataDict2vars(simdataDict)
    

    for s, o in zip([x_sim, y_sim, dur_sim, im_sim], [x, y, dur_dat, im_dat]):
        assert len(s) == len(o)
        assert type(s) == type(o)
        testsub = np.random.randint(len(o))
        assert len(s[testsub]) == len(o[testsub])
        assert type(s[testsub]) == type(o[testsub])
        testtrial = np.random.randint(len(o[testsub]))
        assert len(s[testsub][testtrial]) == len(o[testsub][testtrial])
        assert type(s[testsub][testtrial]) == type(o[testsub][testtrial])
    shutil.rmtree(folderPath)