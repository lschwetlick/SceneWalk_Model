"""
tests for simulation module
"""
from collections import OrderedDict
import os
import shutil
import numpy as np
from scenewalk.simulation import simulate_dataset as sim
from scenewalk.scenewalk_model_object import scenewalk as scenewalk_obj


def test_sample_from_chains_dict():
    vp_params = np.load("tests/test_estimation/sample_dict.npy").item()
    testsub = np.random.randint(0, len(vp_params.keys()))
    param_dict = sim._sample_from_chains_dict(vp_params[testsub])
    assert all([isinstance(v, (int, float, np.int64)) for v in param_dict.values()])
    print()
    for p in param_dict:
        if not isinstance(vp_params[testsub][p], (list, np.ndarray)):
            assert param_dict[p] == vp_params[testsub][p]
        else:
            assert param_dict[p] in vp_params[testsub][p]

def test_simulate_by_subj():
    dur_dat = np.load("tests/test_simdata/sim_dur.npy")
    x = np.load("tests/test_simdata/sim_x.npy")
    y = np.load("tests/test_simdata/sim_y.npy")
    im_dat = np.load("tests/test_simdata/sim_im.npy")
    densities_dat = np.load("tests/test_simdata/densities.npy")
    d_range = np.load("tests/test_simdata/sim_range.npy")

    kwargs_d = {"coupled_oms": True, "coupled_facil":True}
    sw_args = ['subtractive', 'cb','both', 1, 'on', {'x': d_range[0], 'y': d_range[1]}, kwargs_d]
    sw_model = scenewalk_obj(*sw_args)
    params_by_sub = np.load("tests/test_estimation/point_est_subj.npy").item()

    sim_id = sim.simulate(dur_dat, im_dat, densities_dat, sw_model, params = params_by_sub, start_loc="data", x_path=x, y_path=y, resample_durs=True)
    #sim_id = "sim_20191025-101400"
    sim_id = sim_id[4:]
    cwd = os.getcwd()
    folderPath = cwd + "/sim_%s" % sim_id
    x_sim = np.load(folderPath + "/%s_sim_x.npy" % sim_id)
    y_sim = np.load(folderPath + "/%s_sim_y.npy" % sim_id)
    dur_sim = np.load(folderPath + "/%s_sim_dur.npy" % sim_id)
    im_sim = np.load(folderPath + "/%s_sim_im.npy" % sim_id)
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


def test_simulate_sample():
    dur_dat = np.load("tests/test_simdata/sim_dur.npy")
    x = np.load("tests/test_simdata/sim_x.npy")
    y = np.load("tests/test_simdata/sim_y.npy")
    im_dat = np.load("tests/test_simdata/sim_im.npy")
    densities_dat = np.load("tests/test_simdata/densities.npy")
    d_range = np.load("tests/test_simdata/sim_range.npy")

    kwargs_d = {"coupled_oms": True, "coupled_facil":True}
    sw_args = ['subtractive', 'cb','both', 1, 'on', {'x': d_range[0], 'y': d_range[1]}, kwargs_d]
    sw_model = scenewalk_obj(*sw_args)
    chains_dict = np.load("tests/test_estimation/sample_dict.npy").item()

    sim_id = sim.simulate_sample(dur_dat, im_dat, densities_dat, sw_model, chains_dict, "trial", start_loc="center", x_path=None, y_path=None, resample_durs=False)
    #sim_id = "sim_20191025-101400"
    sim_id = sim_id[4:]
    cwd = os.getcwd()
    folderPath = cwd + "/sim_%s" % sim_id
    x_sim = np.load(folderPath + "/%s_sim_x.npy" % sim_id)
    y_sim = np.load(folderPath + "/%s_sim_y.npy" % sim_id)
    dur_sim = np.load(folderPath + "/%s_sim_dur.npy" % sim_id)
    im_sim = np.load(folderPath + "/%s_sim_im.npy" % sim_id)
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