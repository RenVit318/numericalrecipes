#################
#
# Functions for statistical tests and likelihoods
# 
#################

def poisson_likelihood():
    """""""
    pass

def compute_chi_sq(x, y, sigma, func, params):
    """Compute the chi squared value between N points x, y with
    y uncertainty sigma and a function func with parameters params
    Setting sigma = 1 reduces this to just lesat squares"""
    return np.sum(((y - func(x, *params))**2)/(sigma*sigma))

def L1(xdata, ydata, sigma, func, params):
    """Returns the negative log likelihood of an L1 estimator"""
    z = (ydata - func(xdata, *params))/sigma
    return np.sum(np.abs(z))

def lorentzian(xdata, ydata, sigma, func, params):
    """Returns the negative log likelihood of a Lorentzian estimator"""
    z = (ydata - func(xdata, *params))/sigma
    return np.sum(np.log(1. + 0.5*z*z))
