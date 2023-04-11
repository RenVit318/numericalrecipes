import numpy as np
import matplotlib.pyplot as plt
from ancillary import golden_section_search, make_bracket

def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def N(x, A, Nsat, a, b, c):
    """Number of satellites at a distance x. This is the
    function n(x, ..) integrated over the full sphere at x"""
    return 4.*np.pi*x*x*n(x, A, Nsat, a, b, c)

def neg_N(x, A, Nsat, a, b, c):
    """-N(x), used to find the maximum of N(x)"""
    return -4.*np.pi*x*x*n(x, A, Nsat, a, b, c)

def maximization():
    """Code for Q1a"""
    # Constants
    a = 2.4     
    b = 0.25
    c = 1.6
    x_min = 1e-4 # this cannot be zero because of the factor (x/b)^(a-3) and a-3 < 0
    x_max = 5
    Nsat = 100
    A = 256./(5.*np.pi**(3./2.))

    # Maximizing a function f is equal to minimizing -f 
    minim_func = lambda x: -1*N(x, A, Nsat, a, b, c)

    # Make a three-point bracket surrounding the minimum. As initial
    # edges we take the edges of the interval [0, 5]
    bracket, _ = make_bracket(minim_func, [x_min, x_max])
    minimum, _ = golden_section_search(minim_func, bracket)
    print(f'Maximum of N(x) found at x = {minimum:.2f}, N(x) = {N(minimum, A, Nsat, a, b, c):.2f}')
    

def chisq_fit():
    """Code for Q1b"""
    pass
def poisson_fit(): 
    """Code for Q1c"""
    pass
def stat_test():
    """Code for Q1d"""
    pass
def mcmc():
    """Code for Q1e"""
    pass

def full_run():
    maximization()
    #chisq_fit()
    #poisson_fit()
    #stat_test()
    #mcmc()

def main():
    full_run()


if __name__ == '__main__':
    main()

