import numpy as np
import glob
import warnings
import re
from random import sample

data_path_dict = {
    "spst_training": '../../../../DATA/SpatStat/split_sets/training/',
    "spst_test": '../../../../DATA/SpatStat/split_sets/test/',

}


def get_set_names():
    """
    get available data sets
    """
    for n in data_path_dict:
        print(n)

def load_data(set_name):
    """
    Returns selected data set as a dictionary
    """
    loaded_dict = {}

    try:
        folder = data_path_dict[set_name]
    except:
        e = sys.exc_info()[0]
        raise (e)

    load_paths = glob.glob(folder + "*.npy")
    for p in load_paths:
        mat = re.search(r'_([a-zA-Z]*).npy', p)
        name = mat.group(1)
        loaded_dict[name] = np.load(p)

    return (loaded_dict)

def dataDict2vars(data_dict):
    """
    takes data dictionary and returns vectors
    """
    return (data_dict["x"],
            data_dict["y"],
            data_dict["dur"],
            data_dict["im"],
            data_dict["densities"],
            data_dict["range"])


def shorten_set(data_dict, nvp, vps = None):
    """
    Gets x subjects from the selected set
    """
    chosen_vps = []
    for i in data_dict:
        if i == "range" or i == "densities":
            continue
        dat = data_dict[i]
        assert nvp <= len(dat)
        if len(chosen_vps) == 0:
            if vps is None:
                chosen_vps = sample(list(range(len(dat))), nvp)
            else:
                chosen_vps = vps
            print("shortening from "+str(len(dat))+" to "+str(nvp))
        data_dict[i] = [dat[vp] for vp in chosen_vps]
    return(data_dict)