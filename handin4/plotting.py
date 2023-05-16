import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_styles():
    """For consistent plotting scheme"""
    plt.style.use('default')
    mpl.rcParams['axes.grid'] = True
    plt.style.use('seaborn-darkgrid')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['font.size'] = 14


def hist(x, binmin, binmax, nbins, log=False, return_centers=False):
    """"""
    if log:
        bin_edges = np.logspace(np.log10(binmin), np.log10(binmax), nbins + 1)
    else:
        bin_edges = np.linspace(binmin, binmax, nbins + 1)
    if return_centers:
        bin_centers = np.zeros(nbins)

    histogram = np.zeros(nbins)
    for i in range(nbins):
        bin_mask = (x >= bin_edges[i]) * (x < bin_edges[i + 1])
        if log:
            histogram[i] = len(x[bin_mask])
        else:
            histogram[i] = len(x[bin_mask])
        if return_centers:
            bin_centers[i] = bin_edges[i] + 0.5 * (bin_edges[i+1] - bin_edges[i])

    if return_centers:
        return histogram, bin_edges, bin_centers
    return histogram, bin_edges
