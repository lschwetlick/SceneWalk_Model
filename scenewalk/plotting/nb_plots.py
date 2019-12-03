from matplotlib import pyplot as plt
import numpy as np

def priors_plot(priors):
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
        ax[idx].set_xlim(m-sd*1, m+sd*1)
        ax[idx].axvline(x=ub, c="r", linestyle=":")
        ax[idx].axvline(x=lb, c="r", linestyle=":")
        idx += 1
    return (fig, ax)
