import os
import sys
import glob
import warnings
import re
import numpy as np
from random import sample
import yaml
from pathlib import Path


def setup_paths():
    calling_path = os.path.abspath(os.getcwd())
    func_path = __file__
    commn = os.path.commonpath([calling_path, func_path])
    return commn

config = yaml.safe_load(open(Path(__file__).parent / "../../config.yml"))
abspath = config["abs_datapath"]
data_path_dict = config["datasets"]

def get_set_names():
    """
    get available data sets
    """
    for n in data_path_dict:
        print(n)

def setup_data_dict():
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

def load_data(set_name):
    """
    Returns selected data set as a dictionary
    """
    loaded_dict = setup_data_dict()

    try:
        folder = data_path_dict[set_name]
    except:
        e = sys.exc_info()[0]
        raise (e)

    load_paths = glob.glob(folder + "*.npy")
    for p in load_paths:
        mat = re.search(r'_([a-zA-Z]*).npy', p)
        name = mat.group(1)
        loaded_dict[name] = np.load(p, allow_pickle=True)
    if all([x is None for x in loaded_dict.values()]):
        warnings.warn("Can't find the files where you are looking. Is the path set up correctly?")
    return (loaded_dict)

def load_sim_data(folder_path):
    """
    Takes the absolute path of a folder of simulated data and loads the contents into a dictionary
    """
    loaded_dict = setup_data_dict()
    load_paths = glob.glob(folder_path + "*.npy")
    #print(load_paths)
    for p in load_paths:
        mat = re.search(r'([a-zA-Z]*).npy', p)
        #print("mat", mat)
        name = mat.group(1)
        loaded_dict[name] = np.load(p, allow_pickle=True)
    if all([x is None for x in loaded_dict.values()]):
        warnings.warn("Can't find the files where you are looking. Is the path set up correctly?")
    return (loaded_dict)


def dataDict2vars(data_dict):
    """
    takes data dictionary and returns vectors
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
        print('Probably your dictionary is not complete. Caught this error: ' + repr(error))
        raise error

def shorten_set(data_dict, nvp, vps = None):
    """
    Gets x subjects from the selected set
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
    return (short_dict)

def get_ix_from_set(data_dict, subj_order=None, trials_order=None):
    """
    Takes a list of indeces for trials and for subjects and shortend/reorders the dataset.
    If an integer is given in place of a list, it will expand into a list of all
    subjects up to that one.
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
    return (short_dict)

def change_resolution(densities, new_size):
    """Changes the resolution of a list of densities"""
    from cv2 import resize, INTER_CUBIC
    result = []
    for d in densities:
        new_d = resize(d, dsize=(new_size, new_size),
                           interpolation=INTER_CUBIC)
        new_d = new_d / new_d.sum()
        result.append(new_d)
    return result

def chop_scanpaths(lower, upper, datadict):
    chop_dict = {}
    for i in datadict:
        if i == "range" or i == "densities" or i == "meta":
            chop_dict[i] = datadict[i]
            continue
        chop_dict[i] = chop_list(lower, upper, datadict[i])
    return chop_dict

def chop_list(lower, upper, li):
    allim = []
    for i in li:
        im = []
        for ix, j in enumerate(i):
            if len(j)>upper:
                j = j[:upper]
            if len(j)>lower:
                im.append(j)
        allim.append(im)
    return allim