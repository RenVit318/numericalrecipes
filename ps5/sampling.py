import numpy as np
import matplotlib.pyplot as plt
from rng import rng_from_mwc



def rejection_sampling_old(func, rng, N, shift_x=None, shift_y=None):
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
    
def rejection_sampling(func, rng, N, 
                       shift_x=lambda x: x,
                       shift_y=lambda x: x,
                       x0 = 4891653):
    """Sample a distribution using rejection sampling
    Expand documentation!"""
    
    sampled_points = np.zeros(N)
    num_tries = 0 # For performance testing
    for i in range(N):
        not_sampled = True

        # Keep sampling until we find a x,y pair that fits
        while not_sampled:
            numbers, x0 = rng(2, x0=x0, return_laststate=True) # This is now U(0,1)    
             
            x = shift_x(numbers[0])
            y = shift_y(numbers[1])
            num_tries += 1
            #print(x, y, func(x) )
            if y < func(x):
                sampled_points[i] = x
                not_sampled = False

    print(f'Average No. tries: {num_tries/N:.1f}')
    return sampled_points

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
    func = lambda x: x
    shift_x = lambda x: x 
    sampled = rejection_sampling(func, rng_from_mwc, N, shift_x)

    x = np.linspace(0, 1)
    y = x
    
    plt.hist(sampled, density=True, bins=25)
    plt.plot(x, y)
    plt.show()

def main():
    test_sampling()

if __name__ == '__main__':
    main()
