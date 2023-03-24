import numpy as np 
import matplotlib.pyplot

def hist(x, binmin, binmax, nbins, log=False):
    """"""

    if log:
        bin_edges = np.logspace(np.log10(binmin), np.log10(binmax), nbins+1)
    else:
        bin_edges = np.linspace(binmin, binmax, nbins+1)
    
    histogram = np.zeros(nbins)
    for i in range(nbins):
        bin_mask = (x>bin_edges[i]) * (x<bin_edges[i+1])
        histogram[i] = len(x[bin_mask])

    return histogram, bin_edges
    
