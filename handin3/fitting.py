import numpy as np
from ancillary import romberg_integration
from lm_method import levenberg_marquardt
from ancillary import merge_sort, quasi_newton, downhill_simplex
from plotting import hist

# Main Equations
def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def N(x, A, Nsat, a, b, c):
    """Number of satellites at a distance x. This is the
    function n(x, ..) integrated over the full sphere at x"""
    return 4.*np.pi*x*x*n(x, A, Nsat, a, b, c)

   
## CHI SQUARED FITTING ##
def compute_mean_satellites(x, a, b, c, bin_edges, Nsat):
    """Chi squared function specifically for the distribution of satellite
    galaxies around a massive central, n(x, ..) which we attempt to fit 
    using the assumption of Poisson variance \sigma^2 = \mu. Therefore
    sigma is not used, but we need to pass it for function interoperability"""
    # mean = variance = int(N(x))dx over the bin i
    N_fit = lambda x: N(x, 1, Nsat, a, b, c)
    integral = romberg_integration(N_fit, bin_edges[0], bin_edges[-1], 6)
    A = Nsat/integral
    N_fit = lambda x: N(x, A, Nsat, a, b, c)
    
    if len(x) == 1: # Evaluate only at a single data point
        # the np.min clause is to ensure we never get an index errors
        bin_idx = np.min([np.argmin(bin_edges-x), len(bin_edges)]) 
        return romberg_integration(N_fit, bin_edges[bin_idx], bin_edges[bin_idx+1], 6)
           
    mean_ar = np.zeros(len(bin_edges)-1)
    for i in range(len(mean_ar)):  
        mean_ar[i] = romberg_integration(N_fit, bin_edges[i], bin_edges[i+1], 6)

    return mean_ar

def fit_procedure(x, Nsat, a, b, c, xmin, xmax):
    """Transforms N into a function we can fit where we iteratively compute A. 
    The calculation of A is pulled from handin 2"""
    N_fit = lambda x: N(x, 1, Nsat, a, b, c)
    integral = romberg_integration(N_fit, xmin, xmax, 10)
    A = Nsat/integral
    N_fit = lambda x: N(x, A, Nsat, a, b, c)
    return N(x, A, Nsat, a, b, c)

def fit_satellite_data_chisq(bin_centers, n, Nsat, guess, bin_edges):
    """Function applying the Levenberg-Marquadt algorithm to implement the
    'easy' fit to the data, with some slight modifications"""
    # Need to add in the lambda function so we can pass in the bin_edges we found

    mean_func = lambda x, a, b, c: compute_mean_satellites(x, a, b, c, bin_edges, Nsat)

    # Fit 'fit_func' to the data using a minimiztation of chi^2 defined by chisq_func. It doesn't matter what values
    # we use for sigma, because we will never use it. We set it to 0 here to ensure it's never used
    return levenberg_marquardt(bin_centers, n, None, mean_func, guess, linear=False,
                               chisq_like_poisson=True)

    

## POISSON FITTING ##
 

def poisson_fit_func(x, y, delta_x, bin_edges, Nsat, params, xmin, xmax, no_bins):
    """Procedure to be iteratively called in quasi_newton to fit a Poisson
    distribution to the satellite galaxy data. Adjusted from the chi^2 fit
    for optimization reasons. delta_x is the smallest non-zero difference
    between two x used as a proxy for bin size"""
    # Start by computing A corresponding to these a, b, c like handin 2
    
    N_fit = lambda x: N(x, 1, Nsat, *params)
    integral = romberg_integration(N_fit, xmin, xmax, 8)
    A = Nsat/integral
    N_fit = lambda x: N(x, A, Nsat, *params)

    mean_ar = np.zeros(len(x)) 
    for i in range(len(mean_ar)):
        if no_bins:
            int_min, int_max = x[i] - delta_x, x[i]+delta_x
        else:
            int_min, int_max = bin_edges[i], bin_edges[i+1]  
            
        mean_ar[i] = romberg_integration(N_fit, int_min, int_max, 7)

    if no_bins:
        ll = -1.*np.sum(np.log(mean_ar)) # plus an integral we take as constant
    else:
        ll = -1.*np.sum(y*np.log(mean_ar) - mean_ar) # plus a factor of y! which is constant

    return ll



def fit_satellite_data_poisson(x, y, Nsat, guess, bin_edges, no_bins):
    """Computes the Poisson likelihood of a function in the limit
    where the data is essentially unbinned. This function automatically
    computes the binsize required to obtain this.
    mean_func should be the function that computes the mean of the
    distribution, which should of course be linked to the fit function"""

    if no_bins:
        # Find the smallest difference between two x neighbouring x. Need to sort the 
        # array first to find this. Sorting immediately gives us max(x) and min(x) as well
        # NOTE the below could have been done a lot quicker with numpy!
        x_sorted = merge_sort(x) 
        diff_x = x_sorted[1:] - x_sorted[:-1]
        diff_sorted = merge_sort(diff_x) 
        smallest_diff = diff_sorted[diff_sorted>0][0] # smallest non-zero element
        # This smallest difference sets the "bin size"
    else:
        smallest_diff = None

    # Fitting function and procedure
    fit_func = lambda p: poisson_fit_func(x, y, smallest_diff, bin_edges, Nsat, p, bin_edges[0], bin_edges[-1], no_bins)
    #fit_params, n_iter = quasi_newton(fit_func, guess)
    fit_params, n_iter = downhill_simplex(fit_func, guess)
    logL = fit_func(fit_params)
    return fit_params, logL, n_iter
    
