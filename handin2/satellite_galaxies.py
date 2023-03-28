import numpy as np
import matplotlib.pyplot as plt
from ancillary import romberg_integration, rejection_sampling, ridders_method, merge_sort
from rng import rng_from_mwc
from plotting import hist, set_styles
import time


def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of 
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def analytical_derivative(x, A, Nsat, a, b, c):
    """Analytic derivative of n(x) given above. Calculated by hand, with the help
    of a derivative calculator (derivative-calculator.net)"""
    return -A*Nsat*b*b*b*((x/b)**a)*(c*((x/b)**c) - a + 3.) * np.exp(-(x/b)**c) / (x*x*x*x)

def full_run():
    t0 = time.time()
    # Parameters given in the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    A = 1 # Need to compute this later
    Nsat = 100
    x_min = 0
    x_max = 5
    N_sample = int(1e4)
    sample_min_x = 1e-4
    nbins = 20
    num_random_sample = 100 
    
    results_txt = "This file contains all results to be extracted for the report\n"
    #######
    set_styles()

    # 1a. Integrate the 3D spherical integral to find A
    to_integrate = lambda x: 4*np.pi*x*x* n(x, A, Nsat, a, b, c)
    volume_integral = romberg_integration(to_integrate, x_min, x_max, 10, open_formula=True)
    A = Nsat/volume_integral
    print(f'The volume integral evaluated from 0 to 5 returns: {volume_integral:.6f}')
    print(f'Therefore we need a normalization constant A = {A:.2f}')
    results_txt += f"{A}\n"

    # 1b. Simulate the distribution
    rng = rng_from_mwc
    log_distribution = lambda x: np.log10(4*np.pi*n(10**x, A, Nsat, a, b, c)/Nsat) #p(x)dx = N(x)dx/Nsat
   
    log_ymax = log_distribution(np.log10(sample_min_x))
    log_ymin = log_distribution(np.log10(x_max))
    shift_x = lambda x: np.log10(x * (x_max - sample_min_x) + sample_min_x)
    shift_y = lambda y: y * (log_ymax - log_ymin) + log_ymin  
    
    print('Sampling Distribution')
    sampled_points = rejection_sampling(log_distribution, rng, N_sample, shift_x=shift_x, shift_y=shift_y)
    sampled_points = 10**sampled_points

    # Plotting
    bin_heights, bin_edges = hist(sampled_points, sample_min_x, x_max, nbins, log=True)
    bin_centers = np.zeros(len(bin_edges)-1)
    for i in range(bin_centers.shape[0]):
        bin_centers[i] = bin_edges[i] + 0.5*(bin_edges[i+1] - bin_edges[i])

    x = np.linspace(sample_min_x, x_max, 1000)
    y = log_distribution(np.log10(x))

    plt.step(np.log10(bin_centers), np.log10(bin_heights/100), label='Sampled Distribution')
    #plt.step(bin_centers, bin_heights/100, label='Sampled Distribution')           
    plt.plot(np.log10(x), y, label='Analytical Function')
    plt.xlabel(r'$^{10}\log~r/r_{vir}$')
    plt.ylabel(r'$^{10}\log~p(x)dx$')
    plt.title('Distribution of Satellite Galaxies around a Massive Central')

    plt.ylim(bottom=-5)

    plt.legend()
    plt.savefig('results/satellite_galaxies_pdf.png', bbox_inches='tight')
    plt.clf()

    # 1c. 
    # Step 1: Order a list of random numbers, select the first 100
    print('Shuffling Sample..')
    random_keys = rng(N=N_sample)
    random_idxs = merge_sort(key=random_keys)
    print(random_idxs)
    
    random_sample_idxs = random_idxs[:num_random_sample]
    random_sample = sampled_points[random_sample_idxs]
    print(random_sample)
    # Step 2: Order these 100 galaxies by radius
    random_sample_ordered_idxs = merge_sort(key=random_sample)
    random_sample_ordered = random_sample[random_sample_ordered_idxs]
    print(random_sample_ordered)
    # 'Cumulative Distribtuion Function'
    y = np.arange(num_random_sample)/num_random_sample

    # Add a final point to complete the CDF to (xmax, 1)
    y = np.append(y, 1)
    random_sample_ordered = np.append(random_sample_ordered, x_max)

    plt.plot(random_sample_ordered, y)
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1.1)
    plt.xlabel(r'$r/r_{vir}$')
    plt.ylabel('Cumulative Percentage')
    plt.title(f'CDF of Satellite Galaxies (N = {num_random_sample})') 
    plt.savefig('results/satellite_galaxies_cdf')
    #plt.show()


    # 1d. 
    to_diff = lambda x: n(x, A, Nsat, a, b, c)
    x = [1]
    dndx, diff_unc = ridders_method(to_diff, x, 0.1, 2, 1e-12)
    dndx_analytical = analytical_derivative(x[0], A, Nsat, a, b, c)

    print(f'The derivative of n(x) calculated using Ridders Method at x = {x[0]} is {dndx[0]:.12f} +/- {diff_unc[0]:.3E}')
    print(f'The analytical derivative of n(x) at x = {x[0]} is {dndx_analytical:.12f}')

    results_txt += f'{dndx[0]:.12f}\n{diff_unc[0]:.3E}\n{dndx_analytical:.12f}'
    
    with open('results/satellite_galaxies_results.txt', 'w') as file:
        file.write(results_txt)

    print(f'total time with rejection sampling: {time.time()-t0}s')
    
    

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
