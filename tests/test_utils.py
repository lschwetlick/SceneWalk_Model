from collections import OrderedDict
import numpy as np
from scenewalk.utils import utils
from scenewalk.utils import loadData

def test_save2dict_by_subj():
    # setup a pretend data set
    dataDict = loadData.load_sim_data('tests/test_simdata/')
    x, y, dur, im, densities, _ = loadData.dataDict2vars(dataDict)

    def_args = OrderedDict({
        'omegaAttention': None,
        'omfrac': 10,
        'sigmaAttention': None,
        "sigmaInhib" : None,
        'gamma': None,
        #"lamb":None,
        'inhibStrength': 0.3,
        'zeta': None,
        'sigmaShift': None,
        'shift_size':None,
        'first_fix_OmegaAttention':1.5,
        'cb_sd_x':4,
        'cb_sd_y':3,
        'omega_prevloc_frac':10,
    })
    all_vp_list = list(range(len(im)))
    chains_list = [np.random.rand(3, 500, 7) for v in all_vp_list]
    utils.save2dict_by_subj(chains_list, all_vp_list, def_args, "tests/test_estimation/sample_dict.npy")

    vp_params = np.load("tests/test_estimation/sample_dict.npy", allow_pickle=True).item()
    assert isinstance(vp_params, dict)
    assert list(vp_params.keys()) == all_vp_list
    testsub = np.random.randint(0, len(all_vp_list))
    assert vp_params[testsub].keys() == def_args.keys()
    print(vp_params[testsub].values())
    assert any([isinstance(v, (list, np.ndarray)) for v in vp_params[testsub].values()])
    assert any([isinstance(v, (int, float)) for v in vp_params[testsub].values()])


def test_save2npy_point_estimate_by_subj():
    # setup a pretend data set
    dataDict = loadData.load_sim_data('tests/test_simdata/')
    x, y, dur, im, _, _ = loadData.dataDict2vars(dataDict)

    def_args = OrderedDict({
        'omegaAttention': None,
        'omfrac': 10,
        'sigmaAttention': None,
        "sigmaInhib" : None,
        'gamma': None,
        #"lamb":None,
        'inhibStrength': 0.3,
        'zeta': None,
        'sigmaShift': None,
        'shift_size':None,
        'first_fix_OmegaAttention':1.5,
        'cb_sd_x':4,
        'cb_sd_y':3,
        'omega_prevloc_frac':10,
    })
    all_vp_list = list(range(len(im)))
    chains_list = [np.random.rand(3, 500, 7) for v in all_vp_list]
    utils.save2npy_point_estimate_by_subj(chains_list, all_vp_list, def_args, 0.8, "tests/test_estimation/point_est_subj.npy", CI=False)
    vp_params = np.load("tests/test_estimation/point_est_subj.npy", allow_pickle=True).item()

    assert isinstance(vp_params, dict)
    assert list(vp_params.keys()) == all_vp_list
    testsub = np.random.randint(0, len(all_vp_list))
    assert vp_params[testsub].keys() == def_args.keys()
    assert all([isinstance(v, (int, float)) for v in vp_params[testsub].values()])
