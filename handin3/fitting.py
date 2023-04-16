import numpy as np
from ancillary import romberg_integration
from lm_method import levenberg_marquardt
from ancillary import merge_sort, quasi_newton
from plotting import hist

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

def poissons_likelihood_nobins(x, y, mean_func, params):
    """Compute the likelihood of a function under a Poisson distribution,
    and provided that each y_i is only 0 or 1"""
    ones_mask = y==1
    ll = -1*np.log(mean_func(params))
    ll += mean_func(params) # expensive and maybe not necessary?
    return ll

def fit_satellite_data_poisson_nobins(x, Nsat, guess):
    """Computes the Poisson likelihood of a function in the limit
    where the data is essentially unbinned. This function automatically
    computes the binsize required to obtain this.
    mean_func should be the function that computes the mean of the
    distribution, which should of course be linked to the fit function"""

    # Find the smallest difference between two x neighbouring x. Need to sort the 
    # array first to find this. Sorting immediately gives us max(x) and min(x) as well
    # NOTE the below could have been done a lot quicker with numpy!
    x_sorted = merge_sort(x)
    diff_x = x_sorted[1:] - x_sorted[:-1]
    smallest_diff = merge_sort(diff_x)[0] # smallest element
    
    # This smallest difference sets the bin size
    nbins = int(np.ceil((x_sorted[-1] - x_sorted[0])/smallest_diff) )
    print(nbins)
    n, bin_edges, bin_centers = hist(x, x_sorted[0], x_sorted[-1], nbins, return_centers=True)
    print(np.max(n))

    # The only 'variable' are the parameters. Make functions to reflect this
    fit_func = lambda x, a, b, c: fit_procedure(x, Nsat, a, b, c, bin_edges[0], bin_edges[-1])
    mean_func = lambda p: compute_mean_satellites(bin_centers, n, None, fit_func, p, bin_edges)
    QN_func = lambda p: poissons_likelihood_nobins(bin_centers, n, mean_func, p)
    fit_params, n_iter = quasi_newton(QN_func, guess)

    return fit_params, n_iter
    
    
    

def compute_mean_satellites(x, y, sigma, func, params, bin_edges):
    """Chi squared function specifically for the distribution of satellite
    galaxies around a massive central, n(x, ..) which we attempt to fit 
    using the assumption of Poisson variance \sigma^2 = \mu. Therefore
    sigma is not used, but we need to pass it for function interoperability"""
    # mean = variance = int(N(x))dx over the bin i
    mean_ar = np.zeros(len(bin_edges)-1)
    for i in range(len(mean_ar)):
        mean_ar[i] = romberg_integration(lambda x: func(x, *params), bin_edges[i], bin_edges[i+1], 5)
    return mean_ar

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

    
