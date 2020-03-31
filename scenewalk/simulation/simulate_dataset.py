"""
Simulate data
"""
import os
import time
import numpy as np
from scipy import stats
from scenewalk.utils import utils

def simulate(dur_dat, im_dat, densities_dat, sw_model, params=None,
             start_loc="center", x_path=None, y_path=None, resample_durs=False,
             verbose=False):
    """
    simulate and save dataset given durations and images

    Parameters
    ----------
    dur_dat : array
        duration vector Subjects[Images[Scanpath[]]]
    im_dat : array
        image numbers vector (starts at 1)
        - densities_dat : 128x128px densities in the order of number reference
    sw_model : scenewalk model object
        scenewalk object (if using only one param combination for the whole
        simulation, pass object with preset param values)
    params : {None, dict}
        if None, just use whatever is in the scenewalk object.
        Otherwise pass a dictionary of all the parameters your model needs
    start_loc  : {"center" or "data"}
        start location of each trial
    x_path : {None, Array}
        if start_loc is set to data, provide x positions of the data
    y_path: {None, Array}
        if start_loc is set to data, provide y positions of the data
    resample_durs : bool
        set to true if you want durations to be sampled from a fitted gamma
        distribution for each subject. If False, it uses empirical fixation
        durations.
    verbose : bool
        set to true for basic progress printing
    Returns
    -------
    str
        simulation id number
    """

    data_list_x = []
    data_list_y = []
    data_list_dur = []
    data_list_im = []

    # Find All VP's we need to simulate for
    if params is not None:
        available_vps = params.keys()
    else:
        available_vps = list(range(len(im_dat)))

    sub_cnt = -1
    for sub_dur_dat, sub_im_dat in zip(dur_dat, im_dat): # iterate subjects
        sub_cnt += 1
        if sub_cnt not in available_vps:
            print("not available")
            data_list_x.append(None)
            data_list_y.append(None)
            data_list_dur.append(None)
            data_list_im.append(None)
            continue
        if verbose:
            print("sub", sub_cnt)

        # if we have a seperate fit per vp, we update the model
        if params is not None:
            sub_params = params[sub_cnt]
            sw_model.update_params(sub_params)

        # if we are using gamma sampled durations, figure out the parameters
        if resample_durs:
            sub_durs_flat = \
                [item for sublist in sub_dur_dat for item in sublist]
            gamma = stats.gamma
            x = np.linspace(0, 5, 200)
            dur_param = gamma.fit(sub_durs_flat, floc=0)

        im_list_x = []
        im_list_y = []
        im_list_dur = []
        im_list_im = []

        tr_cnt = -1
        for durations, tr_im in zip(sub_dur_dat, sub_im_dat): # iterate trials
            tr_cnt += 1

            # if we are sampling durations, do so now
            if resample_durs:
                sampled_durations = gamma.rvs(*dur_param, size=len(durations))
                use_durations = sampled_durations
            else:
                use_durations = durations

            # pick starting position
            start_loc_x, start_loc_y = _pick_start_position(start_loc, sw_model,
                                                            x_path, y_path,
                                                            sub_cnt, tr_cnt)

            x, y = sw_model.simulate_scanpath(use_durations,
                                              densities_dat[tr_im[0] - 1],
                                              (start_loc_x, start_loc_y),
                                              get_LL=False)
            im_list_x.append(np.asarray(x))
            im_list_y.append(np.asarray(y))
            im_list_dur.append(np.asarray(use_durations))
            im_list_im.append(np.asarray(tr_im))
        data_list_x.append(im_list_x)
        data_list_y.append(im_list_y)
        data_list_dur.append(im_list_dur)
        data_list_im.append(im_list_im)

    nvp = len(available_vps)
    sim_id = _save_sims(data_list_x, data_list_y, data_list_dur, data_list_im,
                        sw_model, nvp)
    return "sim_%s" % sim_id



def simulate_sample(dur_dat, im_dat, densities_dat, sw_model, chains_dict,
                    sample_level, start_loc="center", x_path=None, y_path=None,
                    resample_durs=False, verbose=False):
    """
    simulates dataset given durations and images but sample from the posterior
    parameter distribution.

    Parameters
    ----------
    dur_dat : array
        duration vector Subjects[Images[Scanpath[]]]
    im_dat : array
        image numbers vector (starts at 1)
        - densities_dat : 128x128px densities in the order of number reference
    sw_model : scenewalk model object
        scenewalk object (if using only one param combination for the whole
        simulation, pass object with preset param values)
    chains_dict : dict
        pass estimated posteriors in a dictionary to sample from
    sample_level : {"vp", "trial", "fix"}
        how often do we sample? every subject, every trial, or every fixation?
    params : {None, dict}
        if None, just use whatever is in the scenewalk object.
        Otherwise pass a dictionary
    start_loc  : {"center" or "data"}
        start location of each trial
    x_path : {None, Array}
        if start_loc is set to data, provide x positions of the data
    y_path: {None, Array}
        if start_loc is set to data, provide y positions of the data
    resample_durs : bool
        set to true if you want durations to be sampled from a fitted gamma
        distribution for each subject. If False, it uses empirical fixation
        durations.
    verbose : bool
        set to true for basic progress printing

    Returns
    -------
    str
        simulation id number
    """
    if sample_level not in ["vp", "trial", "fix"]:
        raise Exception("illegal sample level. must be vp, trial or fix")

    data_list_x = []
    data_list_y = []
    data_list_dur = []
    data_list_im = []

    available_vps = chains_dict.keys()

    sub_cnt = -1
    for sub_dur_dat, sub_im_dat in zip(dur_dat, im_dat): # iterate subjects
        sub_cnt += 1
        if sub_cnt not in available_vps:
            print("not available")
            data_list_x.append(None)
            data_list_y.append(None)
            data_list_dur.append(None)
            data_list_im.append(None)
            continue
        sub_dict = chains_dict[sub_cnt]
        if verbose:
            print("sub", sub_cnt)

        # if we have a seperate fit per vp, we update the model now
        if sample_level == "vp":
            param_dict = {}
            param_dict = _sample_from_chains_dict(sub_dict)
            sw_model.update_params(param_dict)

        # if we are using gamma sampled durations, figure out the parameters
        if resample_durs:
            sub_durs_flat = \
                [item for sublist in sub_dur_dat for item in sublist]
            gamma = stats.gamma
            x = np.linspace(0, 5, 200)
            dur_param = gamma.fit(sub_durs_flat, floc=0)

        im_list_x = []
        im_list_y = []
        im_list_dur = []
        im_list_im = []

        tr_cnt = -1
        for durations, tr_im in zip(sub_dur_dat, sub_im_dat): # iterate trials
            tr_cnt += 1
            if sample_level == "trial":
                param_dict = {}
                param_dict = _sample_from_chains_dict(sub_dict)
                sw_model.update_params(param_dict)
            # if we are sampling durations, do so now
            if resample_durs:
                sampled_durations = gamma.rvs(*dur_param, size=len(durations))
                use_durations = sampled_durations
            else:
                use_durations = durations

            # pick starting position
            start_loc_x, start_loc_y = _pick_start_position(start_loc, sw_model,
                                                            x_path, y_path,
                                                            sub_cnt, tr_cnt)

            x, y = sw_model.simulate_scanpath(use_durations,
                                              densities_dat[tr_im[0] - 1],
                                              (start_loc_x, start_loc_y),
                                              get_LL=False)
            im_list_x.append(np.asarray(x))
            im_list_y.append(np.asarray(y))
            im_list_dur.append(np.asarray(use_durations))
            im_list_im.append(np.asarray(tr_im))
        data_list_x.append(im_list_x)
        data_list_y.append(im_list_y)
        data_list_dur.append(im_list_dur)
        data_list_im.append(im_list_im)

    nvp = len(available_vps)
    sim_id = _save_sims(data_list_x, data_list_y, data_list_dur, data_list_im,
                        sw_model, nvp, sampled=True)
    return "sim_%s" % sim_id


### HELPERS

def _save_sims(data_list_x, data_list_y, data_list_dur, data_list_im, sw_model,
               nvp, sampled=False):
    sim_id = (time.strftime("%Y%m%d-%H%M%S"))
    if sampled:
        sim_id = sim_id + "samp"
    cwd = os.getcwd()
    os.mkdir(cwd + "/sim_%s" % sim_id)
    folderPath = cwd + "/sim_%s" % sim_id
    np.save(folderPath + "/%s_sim_x.npy" % sim_id, data_list_x)
    np.save(folderPath + "/%s_sim_y.npy" % sim_id, data_list_y)
    np.save(folderPath + "/%s_sim_dur.npy" % sim_id, data_list_dur)
    np.save(folderPath + "/%s_sim_im.npy" % sim_id, data_list_im)
    np.save(folderPath + "/%s_sim_range.npy" % sim_id, \
        [sw_model.data_range['x'], sw_model.data_range['y']])
    meta_dict = {
        "model_params": sw_model.get_params(),
        "len": nvp,
        "gitsha": utils.get_git_sha(),
        "sampled": sampled
    }
    np.save(folderPath + "/%s_sim_meta.npy" % sim_id, [meta_dict])
    return sim_id

def _pick_start_position(start_loc, sw_model, x_path, y_path, sub_cnt, tr_cnt):
    if start_loc == "center":
        start_loc_x = sum(sw_model.data_range['x']) / 2
        start_loc_y = sum(sw_model.data_range['y']) / 2
    elif start_loc == "data":
        assert (x_path is not None) & (y_path is not None)
        start_loc_x = x_path[sub_cnt][tr_cnt][0]
        start_loc_y = y_path[sub_cnt][tr_cnt][0]
    else:
        raise Exception("No valid starting method")
    return start_loc_x, start_loc_y

def _sample_from_chains_dict(sub_dict):
    param_dict = {}
    sample_ix = None
    for p in sub_dict:
        if sample_ix is None:
            sample_ix = np.random.randint(0, len(sub_dict[p])-1)
        if isinstance(sub_dict[p], (int, float, np.float, np.int)):
            param_dict[p] = sub_dict[p]
        else:
            param_dict[p] = sub_dict[p][sample_ix]
    return param_dict
