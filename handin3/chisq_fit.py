import numpy as np
from ancillary import romberg_integration
from lm_method import levenberg_marquardt

# Have to import these here again due to circular imports
def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def N(x, A, Nsat, a, b, c):
    """Number of satellites at a distance x. This is the
    function n(x, ..) integrated over the full sphere at x"""
    return 4.*np.pi*x*x*n(x, A, Nsat, a, b, c)
##

def compute_mean_satellites(x, y, sigma, func, params, bin_edges):
    """Chi squared function specifically for the distribution of satellite
    galaxies around a massive central, n(x, ..) which we attempt to fit 
    using the assumption of Poisson variance \sigma^2 = \mu"""
    # mean = variance = int(N(x))dx over the bin i
    mean_ar = np.zeros(len(bin_edges)-1)
    for i in range(len(mean_ar)):
        mean_ar[i] = romberg_integration(lambda x: func(x, *params), bin_edges[i], bin_edges[i+1], 5)
    return np.sqrt(mean_ar)

def fit_procedure(x, Nsat, a, b, c, xmin, xmax):
    """Transforms N into a function we can fit where we iteratively compute A. 
    The calculation of A is pulled from handin 2"""
    N_fit = lambda x: N(x, 1, Nsat, a, b, c)
    integral = romberg_integration(N_fit, xmin, xmax, 10)
    A = Nsat/integral
    
    return N(x, A, Nsat, a, b, c)

def fit_satellite_data_chisq(bin_centers, n, Nsat, guess, bin_edges):
    """Function applying the Levenberg-Marquadt algorithm to implement the
    'easy' fit to the data, with some slight modifications"""
    # Need to add in the lambda function so we can pass in the bin_edges we found
    sigma_func = lambda x, y, sigma, func, params: compute_mean_satellites(x, y, sigma, func, params, bin_edges)
    fit_func = lambda x, a, b, c: fit_procedure(x, Nsat, a, b, c, bin_edges[0], bin_edges[-1])
    # Fit 'fit_func' to the data using a minimiztation of chi^2 defined by chisq_func. It doesn't matter what values
    # we use for sigma, because we will never use it. We set it to 0 here to ensure it's never used
    return levenberg_marquardt(bin_centers, n, 0, fit_func, guess, linear=False,
                               chisq_like_poisson=True, sigma_func=sigma_func)

    
