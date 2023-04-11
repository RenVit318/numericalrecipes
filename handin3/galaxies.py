import numpy as np
import matplotlib.pyplot as plt
from ancillary import golden_section_search, make_bracket
from plotting import set_styles, hist

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


def fit_satellite_data_chisq(radius, nhalo, guess):
    """"""
    Nsat = len(radius)/nhalo
    # Make own routine using levenberg marquadt iteratively?



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
    

def chisq_fit():
    """Code for Q1b"""
    basename = 'data/satgals_m1'
    xmin = 1e-4 # cannot take zero because it messes with log and powers
    xmax = 5
    Nbins = 20
    do_log = True
    guess = [2, 1, 1]

    # do all of the below for each dataset separately
    for i in range(4, 5):
        # First bin the data in log-space, discard first 5 lines because they're just comments
        radius, nhalo = readfile(f'{basename}{i}.txt')
        n, bin_edges = hist(radius, xmin, xmax, Nbins, do_log)
        bin_centers = np.zeros(len(bin_edges)-1)
        for j in range(len(bin_centers)):
            bin_centers[j] = bin_edges[j] + 0.5*(bin_edges[j+1] - bin_edges[j])
        #plt.step(np.log10(bin_centers), np.log10(n/len(radius)), label=i)

        # Start with fitting a chi squared distribution to this
        fit_satellite_data_chisq(radius, nhalo, guess)



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
    chisq_fit()
    #poisson_fit()
    #stat_test()
    #mcmc()

def main():
    full_run()


if __name__ == '__main__':
    main()


