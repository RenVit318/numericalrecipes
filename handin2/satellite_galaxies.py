import numpy as np
import matplotlib.pyplot as plt
from ancillary import romberg_integration, rejection_sampling
from rng import rng_from_mwc


def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of 
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def n_integral(x, A, Nsat, a, b, c):
    """Function within the integral, to be used for an
    integration algorithm"""
    return 4*np.pi* x*x *n(x, A, Nsat, a, b, c)

def full_run():
    # Parameters given in the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    A = 1 # Need to compute this later
    Nsat = 100
    x_min = 0
    x_max = 5
    N_sample = int(1e5)
    #######

    # 1a. Integrate the 3D spherical integral to find A
    to_integrate = lambda x: n_integral(x, A, Nsat, a, b, c)
    volume_integral = romberg_integration(to_integrate, x_min, x_max, 10, open_formula=True)
    A = Nsat/volume_integral

    # 1b. Simulate the distribution
    rng = rng_from_mwc # UPDATE THIS?
    shift_x = lambda x: x*5 # x in range [0, 5]
    shift_y = lambda x: x # y in range{0,1] should be good right?
    
    # TODO: I don't think tihs is the right distribution to use
    distribution = lambda x: n(x, A, Nsat, a, b, c)/Nsat
    # Rejection sampling is extremely slow. Good speed upgrade here
    sampled_points = rejection_sampling(distribution, rng, N_sample, shift_x=shift_x)
    
    # Plotting
    # TODO: Write OWN plotting code
    plt.hist(np.log10(sampled_points), log=True, bins=20, histtype='step')
    x = np.logspace(np.log10(np.min(sampled_points)), np.log10(5))
    y = distribution(x)
    plt.plot(np.log10(x), y)
    plt.show()

    # 1c. 




def plot_distribution(A):
    x = np.linspace(0.1, 5, 100)
    y = n(x, A, 100, 2.4, 0.25, 1.6)
    plt.plot(x, y)


def main():
    full_run()
    #plot_distribution()

if __name__ == '__main__':
    main()
