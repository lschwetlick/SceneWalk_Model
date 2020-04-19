"""Some useful visualization functions for Notebooks"""
from matplotlib import pyplot as plt
import numpy as np

def priors_plot(priors, xlimsd=1, show_bounds=False):
    """
    Makes a simple plot of the priors in the dictionary

    Parameters
    ----------
    Priors : dict
        dictionary of priors where the key is the name and the value is an
        an instance of scipy.stats.truncated normal or some other distribution

    xlimsd : float
        determines how large the xlims are. mean +_ sd*xlimsd

    show_bounds : bool
        if true, makes xlims contingent on bounds, not on mean

    Returns
    -------
    fig, ax
    """
    nrow = 2
    ncol = np.int(np.ceil(len(priors.values())/2))
    fig, ax = plt.subplots(nrow, ncol, figsize=(20, 10))
    ax = ax.ravel()
    idx = 0
    fig.subplots_adjust(hspace=.5, wspace=0.5)

    for prior in priors.values():
        m = prior.kwds['loc']
        sd = prior.kwds['scale']
        lb = (prior.kwds['a']*sd)+m
        ub = (prior.kwds['b']*sd)+m
        x = np.arange(m-sd*2, 1000, 0.001)
        ax[idx].set_title(list(priors.keys())[idx])
        ax[idx].plot(x, prior.pdf(x))
        if not show_bounds:
            ax[idx].set_xlim(m - sd * xlimsd, m + sd * xlimsd)
        else:
            ax[idx].set_xlim(lb - sd * xlimsd, ub + sd * xlimsd)
        ax[idx].axvline(x=ub, c="r", linestyle=":")
        ax[idx].axvline(x=lb, c="r", linestyle=":")
        idx += 1
    if idx < nrow * ncol:
        ax[idx].set_axis_off()
    return (fig, ax)

def plot_path_on_map(dens, x_path, y_path, sw):
    """
    plots a scanpath on top of the underlying image density

    Parameters
    ----------
    dens : array
        empirical densities

    x_path, y_path : arrays
        fixation locations

    sw : scenewalk model object
        scenewalk model object which includes the data range

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.float64(dens), origin="lower")
    x_px = [sw.convert_deg_to_px(x, 'x') for x in x_path]
    y_px = [sw.convert_deg_to_px(y, 'y') for y in y_path]
    ax.plot(x_px, y_px)
    plt.show()
    return (fig, ax)
