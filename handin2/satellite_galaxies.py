import numpy as np
import matplotlib.pyplot as plt
from ancillary import romberg_integration, rejection_sampling, ridders_method
from rng import rng_from_mwc
from plotting import hist


def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of 
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)


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
    to_integrate = lambda x: 4*np.pi*x*x* n(x, A, Nsat, a, b, c)
    volume_integral = romberg_integration(to_integrate, x_min, x_max, 10, open_formula=True)
    A = Nsat/volume_integral
    print(f'The volume integral evaluated from 0 to 5 returns: {volume_integral:.6f}')

    # 1b. Simulate the distribution
    rng = rng_from_mwc # UPDATE THIS?
    distribution = lambda x: 4*np.pi*n(x, A, Nsat, a, b, c)/Nsat
    shift_x = lambda x: x*5 + 1e-4# x in range [0, 5]
    shift_y = lambda x: x * distribution(1e-4) #  y in range[0,1] should be good right?
    

    # Rejection sampling is extremely slow. Good speed upgrade here
    # Rejection sampling doesn't work for this distribution because it is open on one side
    sampled_points = rejection_sampling(distribution, rng, N_sample, shift_x=shift_x)
    
    # Plotting
    # TODO: Write OWN plotting code
    bin_heights, bin_edges = hist(sampled_points, 1e-4, 5, 20, log=True)
    bin_centers = np.zeros(len(bin_edges)-1)
    for i in range(bin_centers.shape[0]):
        bin_centers[i] = bin_edges[i] + 0.5*(bin_edges[i+1] - bin_edges[i])
    x = np.linspace(np.min(sampled_points), 5)
    y = distribution(x)

    plt.step(np.log10(bin_centers), np.log10(bin_heights/100))
    plt.plot(np.log10(x), np.log10(y))
    plt.show()

    # 1c. 
    # Step 1: Order a list of random numbers, select the first 100
    # Step 2: Order these 100 galaxies by luminosity
    # NEED KEY SORTING FOR BOTH!

    # 1d. 
    to_diff = lambda x: n(x, A, Nsat, a, b, c)
    x = [1]
    dndx, diff_unc = ridders_method(to_diff, x, 0.1, 2, 1e-12)

    print(f'The derivative of n(x) calculated using Ridders Method at x = {x[0]} is {dndx[0]:.12f} +/- {diff_unc[0]:.3E}')


def plot_distribution(A):
    x = np.linspace(0.1, 5, 100)
    y = n(x, A, 100, 2.4, 0.25, 1.6)
    plt.plot(x, y)


def main():
    full_run()
    #plot_distribution(10.78)
    #plt.show()

if __name__ == '__main__':
    main()
