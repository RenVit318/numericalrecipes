import numpy as np
import matplotlib.pyplot as plt
from rng import rng_from_mwc



def rejection_sampling(func, rng, N, shift_x=None, shift_y=None):
    """Sample a distribution using rejection sampling
    Expand documentation!"""
    # The first N points are x, the second N points are y
    uniform_dist = rng(2*N)
    uniform_dist /= np.nanmax(uniform_dist)

    # The above is U(0,1), might want to shift it around
    if shift_x is not None:
        uniform_dist[:N] = shift_x(uniform_dist[:N])
    if shift_y is not None:
        uniform_dist[N:] = shift_y(uniform_dist[N:])

    # Test if y < f(x)
    func_vals = func(uniform_dist[:N])
    not_rejected = uniform_dist[N:] < func_vals

    return uniform_dist[:N][not_rejected]
    

def gauss(x, sigma, mu):
    return (1./(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)**2)/(sigma**2))
   
def cauchy(x, gamma, x0):
    return 1./(np.pi*gamma*(1.+((x-x0)/gamma)**2))

def voigt(x, sigma, mu, gamma, x0):
    return gauss(x, sigma, mu)*cauchy(x, gamma, x0)

def test_sampling():
    #### PARAMS
    N = int(1e5)
    ####
    func = lambda x: voigt(x, 1, 0, 1, 0)
    shift_x = lambda x: (x-0.5) * 2
    sampled = rejection_sampling(func, rng_from_mwc, N, shift_x)

    x = np.linspace(-5, 5)
    y = voigt(x, 1, 0, 1, 0)
    
    plt.hist(sampled, density=True, bins=25)
    plt.plot(x, y)
    plt.show()

def main():
    test_sampling()

if __name__ == '__main__':
    main()
