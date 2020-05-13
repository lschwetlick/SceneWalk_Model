"""
Tool to check whether the model is defined for some parameter bounds
"""

import sys
import time
from collections import OrderedDict
import numpy as np
from argparse import ArgumentParser
from scenewalk.scenewalk_model_object import scenewalk as sw_model

EPS = np.finfo(np.float64).eps

bounds_dict = OrderedDict({
    "omegaAttention" : (EPS, 105),
    #"omegaInhib": (EPS, 100000),
    "omfrac": (10, 10),
    "sigmaAttention" : (EPS, 40),
    "sigmaInhib" : (EPS, 40),
    "gamma" : (1, 1),
    "lamb" : (1, 1),
    "inhibStrength" : (0.3, 0.3),
    "zeta" : (-5, 0),
    "sigmaShift" : (2, 2),
    "shift_size" : (EPS, 4),
    "first_fix_OmegaAttention" : (1.5, 1.5),
    "cb_sd_x" : (4, 4),
    "cb_sd_y" : (3, 3),
    "omega_prevloc_frac": (10, 10),
    'chi': (0.06, 0.06),
    'ompfactor': (-0.6, -0.6)
})
durations = ((0.001, 0.001, 0.001), (1, 1, 1), (10, 10, 10))

def main():
    """
    Parse Commandline Arguments
    """
    parser = ArgumentParser()
    parser.add_argument("sw_args", help="npy file of sw args",
                        type=str)
    parser.add_argument("bounds", help="dictionary of bounds to test",
                        type=str)
    parser.add_argument("file",
                        help="output file to save faulty to",
                        type=str)
    args = parser.parse_args()
    sw_args = np.load(args.sw_args, allow_pickle=True)
    bounds_dict = np.load(args.bounds, allow_pickle=True).item()
    file_path = args.file
    all_ok, faulty_perms, sw = benchmark_sw(sw_args, bounds_dict, file_path)
    if all_ok:
        print("The bounds you are testing for parameters: " + str(sw.get_param_list_order()) + " are okay")
    else:
        print("The bounds you are testing for parameters: " + str(sw.get_param_list_order()) + " seem to make the model numerically instable")
        print(faulty_perms)

def benchmark_sw(sw_args, bounds_dict, file_path):
    """
    Tests the model with the given model specs and bounds.
    """
    sw = sw_model(*sw_args)
    str(sw.get_param_list_order())
    #bounds_dict = update_bounds(sw)
    #print(sw_dict)
    red_bd = reduce_bouds_dict(sw, bounds_dict)
    perms = get_perms(red_bd)
    print("begin")

    faulty_perms = []
    all_ok = True
    for p in progressbar(perms):
        sw.update_params(list(p))
        try:
            for dur in durations:
                run_test(sw, dur)
        except AssertionError:
            all_ok = False
            faulty_perms.append(p)
    np.save(file_path, faulty_perms)
    return all_ok, faulty_perms, sw

def run_test(sw, dur):
    """
    runs 2 evaluations of the model
    """
    init_map_att = sw.att_map_init()
    init_map_inhib = sw.initialize_map_unif()
    fix_dens = np.load('../tests/emp_dens.npy', allow_pickle=True)
    durations1 = dur
    x_deg1 = (67, 50, 24)
    y_deg1 = (37, 50, 29)
    att1, inh1, uFinal1, nextF, LL = sw.evolve_maps(durations1, x_deg1, y_deg1, init_map_att, init_map_inhib, fix_dens, 1)

    assert uFinal1.shape == (128, 128)
    assert isinstance(uFinal1.flatten()[0], np.float128)
    assert not np.isnan(uFinal1).any()
    assert not (uFinal1 == 0).all()
    assert not (uFinal1 < 0).any()
    assert np.isclose(np.sum(uFinal1), 1)

    assert att1.shape == (128, 128)
    assert isinstance(att1.flatten()[0], np.float128)
    assert not np.isnan(att1).any()
    assert not (att1 < 0).any()
    assert not (att1 == 0).all()

    assert inh1.shape == (128, 128)
    assert isinstance(inh1.flatten()[0], np.float128)
    assert not np.isnan(inh1).any()
    assert not (inh1 < 0).any()
    assert not (inh1 == 0).all()


    durations2 = dur
    x_deg2 = (50, 24, 56)
    y_deg2 = (50, 29, 66)
    att2, inh2, uFinal2, nextF, LL = sw.evolve_maps(durations2, x_deg2, y_deg2, att1, inh1, fix_dens, 2)

    assert uFinal2.shape == (128, 128)
    assert isinstance(uFinal2.flatten()[0], np.float128)
    assert not np.isnan(uFinal2).any()
    assert not (uFinal2 == 0).all()
    assert not (uFinal2 < 0).any()
    assert np.isclose(np.sum(uFinal2), 1)

    assert att2.shape == (128, 128)
    assert isinstance(att2.flatten()[0], np.float128)
    assert not np.isnan(att2).any()
    assert not (att2 < 0).any()
    assert not (att2 == 0).all()

    assert inh2.shape == (128, 128)
    assert isinstance(inh2.flatten()[0], np.float128)
    assert not np.isnan(inh2).any()
    assert not (inh2 < 0).any()
    assert not (inh2 == 0).all()

def update_bounds(sw):
    """
    Not useful. Maybe implement smartly moving the bounds to find a good
    compromise
    """
    bounds_dict = {}
    for p in sw.get_param_list_order():
        bounds_dict[p] = [0.1, 1.1]
    return(bounds_dict)

def reduce_bouds_dict(sw, bounds_dict):
    """
    Takes a bounds dict and shortens it to include only those parameters the
    model will accept
    """
    print(sw.get_param_list_order())
    red_bounds_dict = {}
    for p in sw.get_param_list_order():
        red_bounds_dict[p] = bounds_dict[p]
    return red_bounds_dict

def get_perms(bounds_dict):
    """
    gives all permutations of the upper and lower bounds
    """
    p_bounds = bounds_dict.values()
    param_perm = np.array(np.meshgrid(*p_bounds)).T.reshape(-1,len(p_bounds))
    return param_perm

def progressbar(it, prefix="", size=60, file=sys.stdout):
    """
    Shows a progress bar in the terminal
    """
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

if __name__ == "__main__":
    main()