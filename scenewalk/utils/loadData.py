"""Funtions that load, handle and change data"""
import os
import sys
import glob
import warnings
import re
from pathlib import Path
from random import sample
import numpy as np
import yaml

DATA_PATH = None
data_path_dict = {}

def find_data():
    """
    tries to find where your data is hiding. Order of preference:
    1. You've set the DATA_PATH variable
    2. config in code directory
    3. config in working directory
    4. folder called DATA in your working directory
    """
    global data_path_dict
    # first check if the path has been set already
    if DATA_PATH is None:
        # if not, see if you have a config file
        if os.path.exists(Path(__file__).parent / "../../config.yml"):
            from_where = "config in code"
            #print("there is a config")
            populate_path_dict_from_yml(Path(__file__).parent / "../../config.yml")
        elif os.path.exists(os.path.abspath("config.yml")):
            #print("there is a config in wd")
            from_where = "config in wd"
            populate_path_dict_from_yml(os.path.abspath("config.yml"))
        else:
            # if not, check if there is a "DATA folder right there"
            if os.path.exists(os.path.abspath("DATA")):
                from_where = "DATA in wd"
                #print("there is a DATA folder in wd")
                autopopulate_data_path_dict(os.path.abspath("DATA"))
            else:
                raise Exception("You have not set up a folder from which to load data."
                                "at the top of your script please `import scenewalk`"
                                "and then set `scenewalk.DATA_PATH='my/path/to/data'`")
    else:
        from_where = "set DATA_PATH"
        #print("there is a DATA_PATH specific")
        autopopulate_data_path_dict(os.path.abspath(DATA_PATH))
    return from_where

def autopopulate_data_path_dict(folder_path):
    """ finds data paths automatically assuming specific Data structure"""
    global data_path_dict
    load_paths = glob.glob(os.path.join(folder_path, "*/npy/"))
    for npy_path in load_paths:
        dataset_name = npy_path.split("/")[-3]
        data_path_dict[dataset_name] = npy_path
        culprit = check_npy_folder_complete(npy_path)
        if not culprit is None:
            warnings.warn("found data at " + npy_path + " but it is incomplete"
                          "you are missing at least the *" + culprit + "* "
                          "file.")

def check_npy_folder_complete(npy_path):
    """ checks npy data folder is complete"""
    try:
        culprit = "dur"
        assert len(glob.glob(os.path.join(npy_path, '*_dur.npy'))) != 0
        culprit = "x"
        assert len(glob.glob(os.path.join(npy_path, '*_x.npy'))) != 0
        culprit = "y"
        assert len(glob.glob(os.path.join(npy_path, '*_y.npy'))) != 0
        culprit = "im"
        assert len(glob.glob(os.path.join(npy_path, '*_im.npy'))) != 0
        culprit = "densities"
        assert len(glob.glob(os.path.join(npy_path, '*_densities.npy'))) != 0
        culprit = "range"
        assert len(glob.glob(os.path.join(npy_path, '*_range.npy'))) != 0
    except AssertionError:
        return culprit
    return None

def populate_path_dict_from_yml(yml_path):
    """ loads config yml """
    global data_path_dict
    config = yaml.safe_load(open(yml_path))
    data_path_dict = config["datasets"]
    for f in data_path_dict:
        culprit = check_npy_folder_complete(data_path_dict[f])
        if not culprit is None:
            warnings.warn("found folder " + data_path_dict[f] + " but it is incomplete."
                          "You are missing at least the *" + culprit + "* "
                          "file.")


def get_set_names():
    """
    get available data sets

    Returns
    -------
    list
        list of dataset names
    """
    find_data()
    for n in data_path_dict:
        print(n)

def setup_data_dict():
    """
    Helper function that sets up data dict structure

    Returns
    -------
    dict
        dict of all relevant keys for data, but populated with Nones
    """
    data_dict = {
        "x": None,
        "y": None,
        "dur": None,
        "im": None,
        "densities": None,
        "range": None,
        "meta": None
    }
    return data_dict

def load_data(data_ref):
    """
    Returns selected data set as a dictionary

    Parameters
    ----------
    data_ref : str
        name  or path of the set you want to load

    Returns
    -------
    dict
        data dictionary
    """
    find_data()
    loaded_dict = setup_data_dict()

    if data_ref in data_path_dict.keys():
        folder = data_path_dict[data_ref]
    else:
        folder = data_ref

    load_paths = glob.glob(os.path.join(folder, "*.npy"))
    for p in load_paths:
        mat = re.search(r'_([a-zA-Z]*).npy', p)
        name = mat.group(1)
        loaded_dict[name] = np.load(p, allow_pickle=True)
    if all([x is None for x in loaded_dict.values()]):
        warnings.warn("You are looking for data at the following path '"
                      + folder
                      +"'. I can't find the files where you are looking. Is the"
                       " path set up correctly?")
    return loaded_dict

def load_sim_data(folder_path):
    """
    Takes the absolute path of a folder of simulated data and loads the contents
    into a dictionary

    Parameters
    ----------
    folder_path : str
        absolute path of the simulated data folder

    Returns
    -------
    dict
        data dictionary
    """
    loaded_dict = setup_data_dict()
    load_paths = glob.glob(os.path.join(folder_path, "*.npy"))
    #print(load_paths)
    for p in load_paths:
        mat = re.search(r'([a-zA-Z]*).npy', p)
        #print("mat", mat)
        name = mat.group(1)
        loaded_dict[name] = np.load(p, allow_pickle=True)
    if all([x is None for x in loaded_dict.values()]):
        warnings.warn("Can't find the files where you are looking. "
                      "Is the path set up correctly?")
    return loaded_dict


def dataDict2vars(data_dict):
    """
    takes data dictionary and returns vectors
    x, y, dur, im, densities, range
    x_dat, y_dat, dur_dat, im_dat, densities_dat, d_range

    Parameters
    ----------
    data_dict : dict
        data dictionary

    Returns
    -------
    arrays
        x, y, dur, im, densities, range
    """
    try:
        return (data_dict["x"],
                data_dict["y"],
                data_dict["dur"],
                data_dict["im"],
                data_dict["densities"],
                data_dict["range"])
    except Exception as error:
        print('Probably your dictionary is not complete. '
              'Caught this error: ' + repr(error))
        raise error

def shorten_set(data_dict, nvp, vps=None):
    """
    Gets x subjects from the selected set

    Parameters
    ----------
    data_dict : dict
        data dictionary
    nvp : int
        number of subjects to return
    vps : list
        list of vp numbers

    Returns
    -------
    dict
        data dictionary
    """
    short_dict = {}
    chosen_vps = []
    for i in data_dict:
        if i == "range" or i == "densities" or i == "meta":
            short_dict[i] = data_dict[i]
            continue
        dat = data_dict[i]
        assert nvp <= len(dat)
        if len(chosen_vps) == 0:
            if vps is None:
                chosen_vps = sample(list(range(len(dat))), nvp)
            else:
                chosen_vps = vps
            print("shortening from "+str(len(dat))+" to "+str(nvp))
        short_dict[i] = [dat[vp] for vp in chosen_vps]
    return short_dict

def get_ix_from_set(data_dict, subj_order=None, trials_order=None):
    """
    Takes a list of indeces for trials and for subjects and shortend/reorders
    the dataset. The order of the list IS relevant. No Nones are added, so the
    absolute indexes will change!
    If an integer is given in place of a list, it will expand into a list of all
    subjects up to that one.

    Parameters
    ----------
    data_dict : dict
        data dictionary
    subj_order : {int, list}
        list of subject indexes to return
    trials_order : {int, list}
        list of trial indexes to return

    Returns
    -------
    dict
        data dictionary
    """
    short_dict = {}
    for i in data_dict:
        # print(i)
        if i == "range" or i == "densities" or i == "meta":
            short_dict[i] = data_dict[i]
            continue
        dat = data_dict[i]
        short = []
        if subj_order is None:
            subj_order = list(range(len(dat)))
        elif isinstance(subj_order, int):
            subj_order = list(range(subj_order))

        for sub in subj_order:
            # print("sub", sub)
            sub_dat = dat[sub].copy()
            # print(len(sub_dat))
            # print(sub_dat)

            if trials_order is None:
                trials_order = list(range(len(sub_dat)))
            elif isinstance(trials_order, int):
                trials_order = list(range(trials_order))

            trial_short = [sub_dat[t] for t in trials_order]
            short.append(trial_short)
        short_dict[i] = short
    return short_dict

def change_resolution(densities, new_size):
    """
    Changes the resolution of a list of densities

    Parameters
    ----------
    densities : list of arrays
        list of empirical fixation densities
    new_size : int
        number of pixels in one direction after resizing

    Returns
    -------
    list
        of densities
    """
    from cv2 import resize, INTER_CUBIC
    result = []
    for d in densities:
        new_d = resize(d, dsize=(new_size, new_size),
                       interpolation=INTER_CUBIC)
        new_d = new_d / new_d.sum()
        result.append(new_d)
    return result

def chop_scanpaths(lower, upper, datadict):
    """
    Cuts off scanpaths in a data set. To make them equal lengths or to get a
    subset of fixations

    Parameters
    ----------
    densities : list of arrays
        list of empirical fixation densities
    new_size : int
        number of pixels in one direction after resizing

    Returns
    -------
    list
        of densities
    """
    chop_dict = {}
    for i in datadict:
        if i == "range" or i == "densities" or i == "meta":
            chop_dict[i] = datadict[i]
            continue
        chop_dict[i] = _chop_list(lower, upper, datadict[i])
    return chop_dict

def _chop_list(lower, upper, li):
    """helper function to cut indices in a list"""
    allim = []
    for i in li:
        im = []
        for ix, j in enumerate(i):
            if len(j) > upper:
                j = j[:upper]
            if len(j) > lower:
                im.append(j)
        allim.append(im)
    return allim
