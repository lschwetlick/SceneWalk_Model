import os
import pytest
import numpy as np
from scenewalk.scenewalk_model_object import scenewalk as sw_model
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.slowtest
@pytest.mark.parametrize('omegaAttention', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('omegaInhib', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('sigmaAttention', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('sigmaInhib', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('gamma', (np.finfo(np.float64).eps, 15))
@pytest.mark.parametrize('lamb', (np.finfo(np.float64).eps, 15))
@pytest.mark.parametrize('inhibStrength', (np.finfo(np.float64).eps, 100000))
@pytest.mark.parametrize('zeta', (np.finfo(np.float64).eps, 1))
@pytest.mark.parametrize('durations', ((0.001, 0.001, 0.001), (1, 1, 1), (10, 10, 10)))
def test_evolve_maps_boundaries(omegaAttention, omegaInhib, sigmaAttention, sigmaInhib, gamma, lamb, inhibStrength, zeta, durations):

    sw_dict = {
        "omegaAttention" : omegaAttention,
        "omegaInhib" :omegaInhib,
        "sigmaAttention" : sigmaAttention,
        "sigmaInhib" : sigmaInhib,
        "gamma" : gamma,
        "lamb" : lamb,
        "inhibStrength" : inhibStrength,
        "zeta" : zeta,
    }

    sw = sw_model("subtractive", "zero", "off", 2, "off", {'x': (23, 100), 'y': (0, 90)})
    param_names = sw.get_param_list_order()
    sw_params = [sw_dict[n] for n in param_names]
    print(sw_params)
    sw.update_params(sw_params)
    sw_params = [omegaAttention, omegaInhib, sigmaAttention, sigmaInhib, gamma, lamb, inhibStrength, zeta]

    init_map_att = sw.att_map_init()
    init_map_inhib = sw.initialize_map_unif()
    fix_dens = np.load(os.path.join(THIS_DIR, 'emp_dens.npy'))
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
