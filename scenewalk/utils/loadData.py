import os
import sys
import glob
import warnings
import re
import numpy as np
from random import sample


def setup_paths():
    calling_path = os.path.abspath(os.getcwd())
    func_path = __file__
    commn = os.path.commonpath([calling_path, func_path])
    return commn

abspath = setup_paths()

data_path_dict = {
    "spst_training": abspath+'/DATA/SpatStat/split_sets/training/',
    "spst_test": abspath+'/DATA/SpatStat/split_sets/test/',
    "corpus_test": abspath+'/DATA/corpusData/split_sets/test/',
    "corpus_training": abspath+'/DATA/corpusData/split_sets/training/',

}


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
    print(load_paths)
    for p in load_paths:
        mat = re.search(r'([a-zA-Z]*).npy', p)
        print("mat", mat)
        name = mat.group(1)
        loaded_dict[name] = np.load(p, allow_pickle=True)
    if all([x is None for x in loaded_dict.values()]):
        warnings.warn("Can't find the files where you are looking. Is the path set up correctly?")
    return (loaded_dict)


def dataDict2vars(data_dict):
    """
    takes data dictionary and returns vectors
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
        if i == "range" or i == "densities":
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

def chop_scanpaths(lower, upper, datadict):
    chop_dict = {}
    for i in datadict:
        if i == "range" or i == "densities":
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