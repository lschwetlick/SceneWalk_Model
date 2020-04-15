"""
Tests for all component functions of the scenewalk object
"""
import pytest
import numpy as np
from scenewalk.scenewalk_model_object import scenewalk as sw_model

@pytest.mark.slowtest
@pytest.mark.parametrize('omegaAttention', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('omegaInhib', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('sigmaAttention', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('sigmaInhib', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('gamma', (np.finfo(np.float64).eps, 7))
@pytest.mark.parametrize('lamb', (np.finfo(np.float64).eps, 7))
@pytest.mark.parametrize('inhibStrength', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('zeta', (np.finfo(np.float64).eps, 1))
@pytest.mark.parametrize('sigmaShift', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('shift_size', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('first_fix_OmegaAttention', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('cb_sd_x', (0.1, 1000))
@pytest.mark.parametrize('cb_sd_y', (0.1, 1000))
@pytest.mark.parametrize('omega_prevloc', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('durations', ((0.001, 0.001, 0.001), (1, 1, 1), (10, 10, 10)))
@pytest.mark.parametrize('inhib_method', ("subtractive", "divisive"))
@pytest.mark.parametrize('att_map_init_type', ("zero", "cb"))
@pytest.mark.parametrize('shifts', ("off", "pre", "post", "both"))
@pytest.mark.parametrize('exponents', (1, 2))
@pytest.mark.parametrize("locdep_decay_switch", ("on", "off"))
def test_evolve_maps_boundaries(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch,
                                omegaAttention, omegaInhib, sigmaAttention, sigmaInhib, gamma, lamb, inhibStrength, zeta,
                                sigmaShift, shift_size, first_fix_OmegaAttention, cb_sd_x, cb_sd_y, omega_prevloc, durations):

    sw_dict = {
        "omegaAttention" : omegaAttention,
        "omegaInhib" :omegaInhib,
        "sigmaAttention" : sigmaAttention,
        "sigmaInhib" : sigmaInhib,
        "gamma" : gamma,
        "lamb" : lamb,
        "inhibStrength" : inhibStrength,
        "zeta" : zeta,
        "sigmaShift" : sigmaShift,
        "shift_size" : shift_size,
        "first_fix_OmegaAttention" : first_fix_OmegaAttention,
        "cb_sd_x" : cb_sd_x,
        "cb_sd_y": cb_sd_y,
        "omega_prevloc" : omega_prevloc,
    }

    sw = sw_model(inhib_method, att_map_init_type, shifts, exponents, locdep_decay_switch, {'x': (23, 100), 'y': (0, 90)})

    sw.update_params(sw_dict)

    init_map_att = sw.att_map_init()
    init_map_inhib = sw.initialize_map_unif()
    fix_dens = np.load('tests/emp_dens.npy')
    durations1 = durations
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


    durations2 = durations
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