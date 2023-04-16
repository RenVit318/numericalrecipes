import numpy as np
import matplotlib.pyplot as plt
from ancillary import golden_section_search, make_bracket, romberg_integration
from plotting import set_styles, hist
from fitting import fit_satellite_data_chisq, fit_procedure, compute_mean_satellites, fit_satellite_data_poisson_nobins

def readfile(filename):
    """Code to read in the halo data, copied from the hand in instructions:
    https://home.strw.leidenuniv.nl/~daalen/Handin_files/satellites2.py"""
    f = open(filename, 'r')
    data = f.readlines()[3:]  # Skip first 3 lines
    nhalo = int(data[0])  # number of halos
    radius = []

    for line in data[1:]:
        if line[:-1] != '#':
            radius.append(float(line.split()[0]))

    radius = np.array(radius, dtype=float)
    f.close()
    return radius, nhalo  # Return the virial radius for all the satellites in the file, and the number of halos

def n(x, A, Nsat, a, b, c):
    """Density profile of the spherical distribution of
    satellite galaxies around a central as a function of
    x = r/r_vir. The values given come from hand-in 2"""
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def N(x, A, Nsat, a, b, c):
    """Number of satellites at a distance x. This is the
    function n(x, ..) integrated over the full sphere at x"""
    return 4.*np.pi*x*x*n(x, A, Nsat, a, b, c)
   

def maximization():
    """Code for Q1a"""
    # Constants
    a = 2.4     
    b = 0.25
    c = 1.6
    xmin = 1e-4 # this cannot be zero because of the factor (x/b)^(a-3) and a-3 < 0
    xmax = 5
    Nsat = 100
    A = 256./(5.*np.pi**(3./2.))

    # Maximizing a function f is equal to minimizing -f 
    minim_func = lambda x: -1*N(x, A, Nsat, a, b, c)

    # Make a three-point bracket surrounding the minimum. As initial
    # edges we take the edges of the interval [0, 5]
    bracket, _ = make_bracket(minim_func, [xmin, xmax])
    x_at_max, _ = golden_section_search(minim_func, bracket)
    max_val = N(x_at_max, A, Nsat, a, b, c)

    print(f'Maximum of N(x) found at x = {x_at_max:.2f}, N(x) = {max_val:.2f}')

    xx = np.linspace(xmin, xmax, 1000)
    yy = N(xx, A, Nsat, a, b ,c)
    plt.plot(xx, yy)
    plt.scatter(x_at_max, max_val, c='red', marker='X', s=100,zorder=3)
    plt.show()
    
def make_plot_alldata():
    """Make a plot showcasing all raw, binned data for the report"""
    basename = 'data/satgals_m1'
    xmin = 1e-4 # cannot take zero because it messes with log and powers
    xmax = 5
    Nbins = 20
    do_log = True
    fig, ax = plt.subplots(1,1)

    for i in range(1, 6):
        radius, nhalo = readfile(f'{basename}{i}.txt')
        n, bin_edges = hist(radius, xmin, xmax, Nbins, do_log)

        n /= nhalo
        bin_centers = np.zeros(len(bin_edges)-1)
        for j in range(len(bin_centers)):
            bin_centers[j] = bin_edges[j] + 0.5*(bin_edges[j+1] - bin_edges[j])

        ax.step(bin_centers, n, label='Data', where='mid')

    xlim, ylim = ax.get_xlim(), ax.get_ylim() # For future plotting of individual sets
    ax.set_xlabel(r'$r/r_{vir}$')
    ax.set_ylabel(r'$N_{sat}(r)/N_{halo}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('results/satellite_data.png', bbox_inches='tight')
    
    return xlim, ylim


def fit_data(xlim, ylim):
    """Code for Q1b-d"""
    basename = 'data/satgals_m1'
    xmin = 1e-4 # cannot take zero because it messes with log and powers
    xmax = 5
    Nbins = 20
    do_log = True
    guess = np.array([2.6, 0.25, 1.25])

    # do all of the below for each dataset separately
#    for i in range(1, 6):
    for i in range(5,6):
        radius, nhalo = readfile(f'{basename}{i}.txt')
        n, bin_edges = hist(radius, xmin, xmax, Nbins, do_log)

        n /= nhalo
        bin_centers = np.zeros(len(bin_edges)-1)
        for j in range(len(bin_centers)):
            bin_centers[j] = bin_edges[j] + 0.5*(bin_edges[j+1] - bin_edges[j])

        # 1b. Start with fitting a chi squared distribution to this using the Levenberg-Marquardt algorithm
        # the biggest adaptation to it is that sigma is iteratively computed such that \sigma^2 = \mu
        Nsat = len(radius)/nhalo
        #params_chi2, chi2, num_iter_chi2 = fit_satellite_data_chisq(bin_centers, n, Nsat, guess, bin_edges)
        params_chi2 = guess
        #print('Chi2 Fit Parameters', fit_params)
        # Bin this chi2 model by integrating over our bins for stat. test and plotting
        xx = np.logspace(np.log10(xmin), np.log10(xmax), 100)
        fit_func = lambda x, a, b, c: fit_procedure(x, Nsat, a, b, c, xmin, xmax)
        chi2_binned = compute_mean_satellites(xx, None, None, fit_func, params_chi2, bin_edges) 
        
        
        # Now fit a Poisson distribution to this data using the Quasi-Newton method
        params_chi2, niter = fit_satellite_data_poisson_nobins(radius, Nsat, guess)
        print('niter', niter)

        # Plotting 
        fig, ax = plt.subplots(1,1)
        ax.step(bin_centers, n, label='Data', where='mid')
        ax.scatter(bin_centers, n, c='black', marker='X', s=50, zorder=5, label='Fit Points')

        
        #xx = np.logspace(np.log10(xmin), np.log10(xmax), 250)        
        #yy = N(xx, A, Nsat, *fit_params)
        yy = fit_procedure(xx, Nsat, *params_chi2, xmin, xmax)
        ax.plot(xx, yy, label=r'$\chi^2$ Fit')
        ax.step(bin_centers, chi2_binned, where='mid')

        #ax.set_title(rf'M = $10^{{{i+10}}} M_{{\odot}}$')


        #plt.legend()
        ax.set_xlabel(r'$r/r_{vir}$')
        ax.set_ylabel(r'$N_{sat}(r)/N_{halo}$')
        #print(xlim, ylim)
        #ax.set_xlim(-0.19, 4)
        #ax.set_ylim(-27, 567)
        ax.set_xscale('log')
        ax.set_yscale('log')
        #plt.savefig(f'results/M1{i}_fit.png', bbox_inches='tight')

        plt.show()


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
    set_styles()
    #maximization()
    #xlim, ylim = make_plot_alldata()
    # Remove the below later!
    xlim, ylim = [-0.19762885165353622, 4.15319535987791], [-27.0271995528402, 567.5711906096442]
    fit_data(xlim, ylim)
    #poisson_fit()
    #stat_test()
    #mcmc()

def main():
    full_run()


if __name__ == '__main__':
    main()


