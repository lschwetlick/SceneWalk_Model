"""
Tests for all component functions of the scenewalk object
"""
import os
import sys
from collections import OrderedDict
import pytest
import numpy as np
from scenewalk.scenewalk_model_object import scenewalk as sw_model

sys.setrecursionlimit(10000)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# def test_check_params_for_config(self):
# def get_param_list_order(self):
# def update_params(self, scene_walk_params):

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

@pytest.mark.basictest
def test_convert_deg_to_px():
    """checks conversion and conversion edge case where fix is outside range"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    deg_val = (23+100)/2
    correct_x_px = 64
    calc_px = sw.convert_deg_to_px(deg_val, 'x', fix=True)
    assert calc_px == correct_x_px
    assert sw.convert_deg_to_px(91, 'y', fix=True) == 127

@pytest.mark.basictest
def test_convert_px_to_deg():
    """checks conversion """
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    deg_val = (23+100)/2
    calc_px = sw.convert_deg_to_px(deg_val, 'x', fix=True)
    calc_deg = sw.convert_px_to_deg(calc_px, 'x')
    assert np.isclose(calc_deg, deg_val)

@pytest.mark.basictest
def test_get_unit_vector():
    """checks unit vector length and magnitude"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    correct_mag = np.sqrt(8)
    correct_u = (np.sqrt(1/2), np.sqrt(1/2))
    u, mag = sw.get_unit_vector((2, 3), (4, 5))
    assert np.isclose(u[0], correct_u[0])
    assert mag == correct_mag

@pytest.mark.basictest
def test_simulate_durations():
    """checks durations have right type and amount"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    n_points = 10
    durs = sw.simulate_durations(n_points)
    assert len(durs) == n_points
    types = [isinstance(x, float) for x in durs]
    assert sum(types) == n_points

@pytest.mark.basictest
def test_empirical_fixation_density():
    """checks fixation density has correct size and type"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    x_locs = [43, 90, 30, 80]
    y_locs = [35, 12, 41, 75]
    x_locs_px, y_locs_px, fix_dens = sw.empirical_fixation_density(x_locs, y_locs)
    assert fix_dens.shape == (128, 128)
    print(type(fix_dens[0][0]))
    assert isinstance(fix_dens[0][0], np.float128)
    assert len(x_locs_px) == len(x_locs) == len(y_locs_px)

@pytest.mark.basictest
def test_fixation_picker_max():
    """check the right el is picked out of matrix"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    mat = np.zeros((128, 128))
    correct_ij = (80, 20)
    mat[correct_ij] = 1
    mat = mat/np.sum(mat)
    x, y = sw.fixation_picker_max(mat)
    j = sw.convert_deg_to_px(x, 'x', fix=True)
    i = sw.convert_deg_to_px(y, 'y', fix=True)
    assert np.isclose(i, correct_ij[0]) and np.isclose(j, correct_ij[1])

@pytest.mark.basictest
def test_fixation_picker_stoch():
    """check the right el is picked out of matrix"""
    import random
    random.seed(1234)
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    mat = np.zeros((128, 128))
    correct_ij = (80, 20)
    mat[correct_ij] = 1
    mat = mat/np.sum(mat)
    # check the right el is picked out of matrix
    x, y = sw.fixation_picker_stoch(mat)
    j = sw.convert_deg_to_px(x, 'x', fix=True)
    i = sw.convert_deg_to_px(y, 'y', fix=True)
    assert np.isclose(i, correct_ij[0]) and np.isclose(j, correct_ij[1])

@pytest.mark.basictest
def test_initialize_map_unif():
    """checks fixation density has correct size and type and value"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    init_map = sw.initialize_map_unif()
    assert init_map.shape == (128, 128)
    assert (init_map.flatten() == sw.EPS).all()
    assert isinstance(init_map.flatten()[0], np.float128)
    assert not np.isnan(init_map).any()

@pytest.mark.basictest
def test_initialize_center_bias():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "cb", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.cb_sd = (8, 5)
    cb_map = sw.initialize_center_bias()
    assert cb_map.shape == (128, 128)
    assert isinstance(cb_map.flatten()[0], np.float128)
    assert not np.isnan(cb_map).any()
    assert not (cb_map < 0).any()
    assert not (cb_map == 0).all()

@pytest.mark.basictest
def test_make_attention():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    #durations = (0.6, 0.6, 0.4)
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    assert att_g.shape == (128, 128)
    assert isinstance(att_g.flatten()[0], np.float128)
    assert not np.isnan(att_g).any()
    assert not (att_g < 0).any()
    assert not (att_g == 0).all()

@pytest.mark.basictest
def test_make_attention_post():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "post", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    #durations = (0.6, 0.6, 0.4)
    att_g = sw.make_attention_gauss_post_shift(x_deg, y_deg)
    assert att_g.shape == (128, 128)
    assert isinstance(att_g.flatten()[0], np.float128)
    assert not np.isnan(att_g).any()
    assert not (att_g < 0).any()
    assert not (att_g == 0).all()

@pytest.mark.basictest
def test_make_attention_post_outside():
    """checks what happens when post attention is outside the grid"""
    sw = sw_model("subtractive", "zero", "post", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw_params2 = OrderedDict({
        "omegaAttention" : 1,
        "omegaInhib": 0.1,
        "omfrac": 2,
        "sigmaAttention" : 5,
        "sigmaInhib" : 4,
        "gamma" : 1,
        "lamb" : 1,
        "inhibStrength" : 0.01,
        "zeta" : 0.01,
        "sigmaShift" : 0.00005,
        "shift_size" : 4,
        "phi" : 100,
        "first_fix_OmegaAttention" : 3,
        "cb_sd_x" : 5,
        "cb_sd_y" : 4,
        "omega_prevloc" : 1
    })

    sw.update_params(sw_params2)
    x_deg = (50, 99, 40)
    y_deg = (50, 89, 40)
    #durations = (0.6, 0.6, 0.4)
    att_g = sw.make_attention_gauss_post_shift(x_deg, y_deg)
    assert att_g.shape == (128, 128)
    assert isinstance(att_g.flatten()[0], np.float128)
    assert not np.isnan(att_g).any()
    assert not (att_g < 0).any()
    assert not (att_g == 0).all()


@pytest.mark.basictest
def test_combine_att_fixdens():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (None, 50, None)
    y_deg = (None, 50, None)
    x_locs = [43, 90, 30, 80]
    y_locs = [35, 12, 41, 75]
    _, _, fix_dens = sw.empirical_fixation_density(x_locs, y_locs)
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    comb_map = sw.combine_att_fixdens(att_g, fix_dens)
    assert comb_map.shape == (128, 128)
    assert isinstance(comb_map.flatten()[0], np.float128)
    assert not np.isnan(comb_map).any()
    assert not (comb_map < 0).any()
    assert not (comb_map == 0).all()

@pytest.mark.basictest
def test_make_inhib_gauss():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    att_g = sw.make_inhib_gauss(x_deg, y_deg)
    assert att_g.shape == (128, 128)
    assert isinstance(att_g.flatten()[0], np.float128)
    assert not np.isnan(att_g).any()
    assert not (att_g < 0).any()
    assert not (att_g == 0).all()

@pytest.mark.basictest
def test_differential_time_basic():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    durations = (0.6, 0.6, 0.4)
    init_map = sw.initialize_map_unif()
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    evolved_map = sw.differential_time_basic(durations[1], att_g, init_map, sw.omegaAttention)
    assert evolved_map.shape == (128, 128)
    assert isinstance(evolved_map.flatten()[0], np.float128)
    assert not np.isnan(evolved_map).any()
    assert not (evolved_map < 0).any()
    assert not (evolved_map == 0).all()

@pytest.mark.basictest
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_differential_time_att(locdep_decay_switch):
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "off", 1, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    durations = (0.6, 0.6, 0.4)
    init_map = sw.initialize_map_unif()
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    evolved_map = sw.differential_time_att(durations[1], att_g, init_map, fixs_x=x_deg, fixs_y=y_deg)
    assert evolved_map.shape == (128, 128)
    assert isinstance(evolved_map.flatten()[0], np.float128)
    assert not np.isnan(evolved_map).any()
    assert not (evolved_map < 0).any()
    assert not (evolved_map == 0).all()


@pytest.mark.basictest
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize('inhib', ("subtractive", "divisive"))
def test_combine(exponents, inhib):
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model(inhib, "zero", "off", exponents, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    durations = (0.6, 0.6, 0.4)
    init_map = sw.initialize_map_unif()
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    evolved_map_att = sw.differential_time_att(durations[1], att_g, init_map, fixs_x=x_deg, fixs_y=y_deg)
    inhib_g = sw.make_inhib_gauss(x_deg, y_deg)
    evolved_map_inhib = sw.differential_time_basic(durations[1], inhib_g, init_map, sw.omegaInhib)
    comb_map = sw.combine(evolved_map_att, evolved_map_inhib)
    assert comb_map.shape == (128, 128)
    assert isinstance(comb_map.flatten()[0], np.float128)
    assert not np.isnan(comb_map).any()
    assert not (comb_map == 0).all()

@pytest.mark.basictest
def test_make_positive():
    """checks fixation density has correct size and type and has legal values"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    durations = (0.6, 0.6, 0.4)
    init_map = sw.initialize_map_unif()
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    evolved_map_att = sw.differential_time_att(durations[1], att_g, init_map, fixs_x=x_deg, fixs_y=y_deg)
    inhib_g = sw.make_inhib_gauss(x_deg, y_deg)
    evolved_map_inhib = sw.differential_time_basic(durations[1], inhib_g, init_map, sw.omegaInhib)
    comb_map = sw.combine(evolved_map_att, evolved_map_inhib)
    u = sw.make_positive(comb_map)
    assert u.shape == (128, 128)
    assert isinstance(u.flatten()[0], np.float128)
    assert not np.isnan(u).any()
    assert not (u == 0).all()
    assert not (u < 0).any()

@pytest.mark.basictest
def test_add_noise():
    """checks fixation density has correct size and type and has legal values and is density"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    x_deg = (67, 50, 24)
    y_deg = (37, 50, 29)
    durations = (0.6, 0.6, 0.4)
    init_map = sw.initialize_map_unif()
    att_g = sw.make_attention_gauss(x_deg, y_deg)
    evolved_map_att = sw.differential_time_att(durations[1], att_g, init_map, fixs_x=x_deg, fixs_y=y_deg)
    inhib_g = sw.make_inhib_gauss(x_deg, y_deg)
    evolved_map_inhib = sw.differential_time_basic(durations[1], inhib_g, init_map, sw.omegaInhib)
    comb_map = sw.combine(evolved_map_att, evolved_map_inhib)
    u = sw.make_positive(comb_map)
    uFinal = sw.add_noise(u)
    assert uFinal.shape == (128, 128)
    assert isinstance(uFinal.flatten()[0], np.float128)
    assert not np.isnan(uFinal).any()
    assert not (uFinal == 0).all()
    assert not (uFinal < 0).any()
    assert np.isclose(np.sum(uFinal), 1)

@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_evolve_maps(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch):
    """
    tests evolve maps function in all different configurations. Checks
    - uFinal is correct shape, type, non-negative, nonzero, non-nan, density
    - attmap is correct shape, type, non-negative, nonzero, non-nan
    - inhmap is correct shape, type, non-negative, nonzero, non-nan
    - next_pos is in legal range and right type
    - LL is not nan
    """
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)

    init_map_att = sw.att_map_init()
    init_map_inhib = sw.initialize_map_unif()
    fix_dens = np.load('tests/emp_dens.npy')
    durations1 = (0.2, 0.3, 0.4)
    x_deg1 = (67, 50, 24)
    y_deg1 = (37, 50, 29)
    att1, inh1, uFinal1, next1, LL1 = sw.evolve_maps(durations1, x_deg1, y_deg1, init_map_att, init_map_inhib, fix_dens, 1, sim=False)
    assert next1[0] == x_deg1[2]
    assert next1[1] == y_deg1[2]

    assert isinstance(LL1, np.float128)
    assert not np.isnan(LL1)

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


    durations2 = (0.3, 0.4, 0.1)
    x_deg2 = (50, 24, 56)
    y_deg2 = (50, 29, 66)
    att2, inh2, uFinal2, next2, LL2 = sw.evolve_maps(durations2, x_deg2, y_deg2, att1, inh1, fix_dens, 2, sim=False)
    assert next2[0] == x_deg2[2]
    assert next2[1] == y_deg2[2]

    assert isinstance(LL2, np.float128)
    assert not np.isnan(LL2)

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

@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_get_scanpath_likelihood(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch):
    """Tests that returned scanpath likelihhods are not nan and have the right type"""
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    fix_dens = np.load('tests/emp_dens.npy')
    x_sim = np.load("tests/test_simdata/sim_x.npy")
    y_sim = np.load("tests/test_simdata/sim_y.npy")
    dur_sim = np.load("tests/test_simdata/sim_dur.npy")

    avg_log_ll = sw.get_scanpath_likelihood(x_sim[0][0], y_sim[0][0], dur_sim[0][0], fix_dens)

    assert isinstance(avg_log_ll, np.float128)
    assert not np.isnan(avg_log_ll)

@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_simulate_scanpath(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch):
    """Tests that simulated scanpaths the right type, range and length"""
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    fix_dens = np.load('tests/emp_dens.npy')
    dur_sim = np.load("tests/test_simdata/sim_dur.npy")
    startpos = (np.sum((23, 100))/2, np.sum((0, 90))/2)
    x_path, y_path, avg_log_ll = sw.simulate_scanpath(dur_sim[0][0], fix_dens, startpos, get_LL=True)
    assert len(x_path) == len(y_path) == len(dur_sim[0][0])
    assert (np.asarray(x_path) <= 100).all()
    assert (np.asarray(x_path) >= 23).all()
    assert (np.asarray(y_path) <= 90).all()
    assert (np.asarray(y_path) >= 0).all()
    assert x_path[0] == startpos[0]
    assert y_path[0] == startpos[1]
    assert isinstance(avg_log_ll, np.float128)


@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
@pytest.mark.parametrize("oms_couple", ("on", "off"))
def test_pass_input_params(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, oms_couple):
    """Tests that the values passed in are correctly assigned to the paramters"""
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    if oms_couple == "on":
        sw.coupled_oms = True

    needed_param_names = sw.get_param_list_order()
    p_list = list(range(len(needed_param_names)))
    sw.update_params(p_list)
    if oms_couple == "on":
        p_list[1] = p_list[0]/p_list[1]
    current = sw.get_params()
    sw.check_params_for_config()
    assert list(current.values()) == p_list


@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
@pytest.mark.parametrize("oms_couple", ("on", "off"))
def test_check_params_in_bounds(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, oms_couple):
    """Tests that the bounds checker reacts appropriately:h"""
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    if oms_couple == "on":
        sw.coupled_oms = True

    z = [5, -1, 0.5]
    exp = [False, False, True]
    for i in range(3):
        params = sw_params.copy()
        params["zeta"] = z[i]
        sw.update_params(params)
        assert sw.check_params_in_bounds() == exp[i]


testcases_test_get_phase_times_both = [(1, (None, 0.5, 0.5), (0, 0.45, 0.05)), # first fixation
                                       (2, (0.5, 0.5, 0.5), (0.05, 0.4, 0.05)), # fixation long enough
                                       (3, (0.5, 0.1, 0.5), (0.05, 0.05, 0)), # fixation too short for all
                                       (4, (0.5, 0.04, 0.5), (0.04, 0, 0)), # fixation too short for main
                                       (1, (0.5, 0.05, 0.5), (0, 0.05, 0))] # first fixation too short
@pytest.mark.basictest
@pytest.mark.parametrize('nth,durs,expected', testcases_test_get_phase_times_both)
def test_get_phase_times_both(nth, durs, expected):
    """Checks that helper function returns the correct phase times in all cases"""
    sw = sw_model("subtractive", "zero", "off", 1, "off", {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)
    sw.tau_post = 0.05
    sw.tau_pre = 0.05
    post, main, pre = sw.get_phase_times_both(nth, durs)
    assert np.isclose((post, main, pre), expected).all()

testcases_test_evolve_maps_durs = [(1, (None, 0.5, 0.5)), # first fixation
                                   (2, (0.5, 0.5, 0.5)), # fixation long enough
                                   (3, (0.5, 0.1, 0.5)), # fixation too short for all
                                   (4, (0.5, 0.04, 0.5)), # fixation too short for main
                                   (1, (0.5, 0.05, 0.5))] # first fixation too short
@pytest.mark.basictest
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
@pytest.mark.parametrize("nth, durs", testcases_test_evolve_maps_durs)
def test_evolve_maps_durs(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, nth, durs):
    """
    tests whether all duration edge cases still produce usefull results with regards to:
    evolve maps function in all different configurations. Checks
    - uFinal is correct shape, type, non-negative, nonzero, non-nan, density
    - attmap is correct shape, type, non-negative, nonzero, non-nan
    - inhmap is correct shape, type, non-negative, nonzero, non-nan
    - next_pos is in legal range and right type
    - LL is not nan
    """
    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})
    sw.update_params(sw_params)

    init_map_att = sw.att_map_init()
    init_map_inhib = sw.initialize_map_unif()
    fix_dens = np.load('tests/emp_dens.npy')
    x_deg1 = (67, 50, 24)
    y_deg1 = (37, 50, 29)
    att1, inh1, uFinal1, next1, LL1 = sw.evolve_maps(durs, x_deg1, y_deg1, init_map_att, init_map_inhib, fix_dens, nth, sim=False)
    assert next1[0] == x_deg1[2]
    assert next1[1] == y_deg1[2]

    assert isinstance(LL1, np.float128)
    assert not np.isnan(LL1)

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
