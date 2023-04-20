import numpy as np
import matplotlib.pyplot as plt
from ancillary import golden_section_search, make_bracket
from plotting import set_styles, hist
from fitting import fit_satellite_data_chisq, compute_mean_satellites, fit_satellite_data_poisson, N
from scipy.special import gammainc, gamma
import time

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

    radius = np.array(radius, dtype=np.float64)
    f.close()
    return radius, nhalo  # Return the virial radius for all the satellites in the file, and the number of halos


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

    print(f'Maximum of N(x) found at x = {x_at_max}, N(x) = {max_val}')

    xx = np.linspace(xmin, xmax, 1000)
    yy = N(xx, A, Nsat, a, b ,c)
    plt.plot(xx, yy)
    plt.scatter(x_at_max, max_val, c='red', marker='+', alpha=0.75, s=100,zorder=3, label=f'Maximum\nx={x_at_max:.2f}\nN(x)={max_val:.2f}')
    plt.axhline(y=0, c='black', ls='--')
    plt.xlim(xmin,xmax)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$N_{sat}(x)$')
    plt.title('Satellite Galaxy Number Distribution')
    plt.legend()
    plt.savefig('results/maxi.png', bbox_inches='tight')

    with open ('results/maxi_results.txt', 'w') as file:
        file.write(f"""Minimization Results
Bracket: {bracket}
x at Max: {x_at_max}
N(x) Max: {max_val}""")
    
    
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

        Nsat = len(radius)/nhalo
        bin_centers = np.zeros(len(bin_edges)-1)
        for j in range(len(bin_centers)):
            bin_centers[j] = bin_edges[j] + 0.5*(bin_edges[j+1] - bin_edges[j])

        ax.step(bin_centers, n, where='mid', label=rf'$M=10^{{{i+10}}}M_{{\odot}}$'+'\n'+rf'$N_{{halo}} = {nhalo}$')#+'\n'+rf'$\left<N_{{sat}}\right>= {Nsat:.3E}$')

    xlim, ylim = ax.get_xlim(), ax.get_ylim() # For future plotting of individual sets
    ax.set_xlabel(r'$r/r_{vir}$')
    ax.set_ylabel(r'$N_{sat}(r)$')
    ax.set_title('Satellite Galaxy Data')

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend(title='Parameters', bbox_to_anchor=(1, 1), frameon=True, fancybox=True)
    plt.savefig('results/satellite_data.png', bbox_inches='tight')
    
    return xlim, ylim


def fit_data():#xlim, ylim):
    """Code for Q1b-d"""
    basename = 'satgals_m1'
    xmin = 1e-4 # cannot take zero because it messes with log and powers
    xmax = 5
    Nbins = 20
    do_log = True
    no_bins = False
    guess = np.array([2.4, 0.25, 1.5])

    fitres_txt = ""
    full_fitres_txt = ""
    stats_txt = ""
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True, figsize=(8, 12))

    # do all of the below for each dataset separately
    for i in range(1,6):
        ax = axs.flatten()[i-1]
        radius, nhalo = readfile(f'{basename}{i}.txt')
        n, bin_edges, bin_centers = hist(radius, xmin, xmax, Nbins, do_log, return_centers=True)

        Nsat = len(radius)/nhalo
        print(f'Imported Dataset M1{i}. {len(radius)} Objects.')

        # 1b. Start with fitting a chi squared distribution to this using the Levenberg-Marquardt algorithm
        # the biggest adaptation to it is that sigma is iteratively computed such that \sigma^2 = \mu
        print('Starting Chi Squared Fitting..')
        params_chi2, chi2, niter_chi2 = fit_satellite_data_chisq(bin_centers, n, Nsat, guess, bin_edges)
        print(f'\nChi Squared Fit:\n\Chi^2 = {chi2}\na, b, c = {params_chi2}\n')

        # 1c. Now fit a Poisson distribution to this data using the Quasi-Newton method
        print('Starting Poisson Fitting..')
        if no_bins:
            params_poisson, logL, niter_poisson = fit_satellite_data_poisson(radius, None, Nsat, guess, bin_edges, no_bins)
        else: # feed it only nbins data points if wanted
            params_poisson, logL, niter_poisson = fit_satellite_data_poisson(bin_centers, n, Nsat, guess, bin_edges, no_bins)
        print(f'\nPoisson Fit:\nlog L = {logL}\n<Nsat> = {Nsat}\na, b, c = {params_poisson[0]}, {params_poisson[1]}, {params_poisson[2]}\n')

        # Bin the Poisson and Chi squared models to match the datA
        xx = np.logspace(np.log10(xmin), np.log10(xmax), 100)
        chi2_binned = nhalo * compute_mean_satellites(xx, *params_chi2, bin_edges, Nsat) 
        poisson_binned = nhalo * compute_mean_satellites(xx, *params_poisson, bin_edges, Nsat) 

        # Statistical Tests
        DoF = Nbins - 4 # degrees of freedom
    
        # G-test for the chi squared model because it is binned
        # mask out all bins without observations: lim_O->0 [O ln(O/E)] = 0 for E != 0
        zero_mask = n != 0
        G_chi2 = 2. * np.sum(n[zero_mask] * np.log((n/chi2_binned)[zero_mask]))
        G_poisson = 2. * np.sum(n[zero_mask] * np.log((n/poisson_binned)[zero_mask]))
        Q_chi2 = (gammainc(DoF/2., G_chi2/2.)/gamma(DoF/2.))
        Q_poisson = (gammainc(DoF/2., G_poisson/2.)/gamma(DoF/2.))
        print(f'G_chi2 = {G_chi2}, G_poisson = {G_poisson}')
        print(f'Q_chi2 = {Q_chi2}, Q_poisson = {Q_poisson}')
         
        # Plotting 
        ax.step(bin_centers, n, label='Data', where='mid')
        ax.scatter(bin_centers, n, c='black', marker='X', s=25, zorder=5, label='Fit Points')
        ax.step(bin_centers, chi2_binned, where='mid', label=r'$\chi^2$ Fit', ls='--')
        ax.step(bin_centers, poisson_binned, where='mid', label='Poisson Fit', ls='--')

        ax.set_title(rf'M = $10^{{{i+10}}} M_{{\odot}}$')      
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(10**-5.5, 10**5.5)
        ax.legend()
              
        fitres_txt += f'$10^{{{i+10}}}$ & {Nsat:.2E} & {params_chi2[0]:.2f} & {params_chi2[1]:.2f} & {params_chi2[2]:.2f} & {chi2:.2E} & \\\\ \n'
        fitres_txt += f' & & {params_poisson[0]:.2f} & {params_poisson[1]:.2f} & {params_poisson[2]:.2f} & & {logL:.2E}\\\\ \n'
        full_fitres_txt += f'$10^{{{i+10}}}$ & {Nsat} & {params_chi2[0]} & {params_chi2[1]} & {params_chi2[2]} & {chi2} & \\\\ \n'
        full_fitres_txt += f' & & {params_poisson[0]} & {params_poisson[1]} & {params_poisson[2]} & & {logL}\\\\ \n'
        stats_txt += f'$10^{{{i+10}}}$ & $\chi^2$ & {G_chi2} & {Q_chi2} \\\\ \n'
        stats_txt += f'$10^{{{i+10}}}$ & Poisson & {G_poisson} & {Q_poisson} \\\\ \n'

    # Figure Labels
    for ax in axs[:,0]:
        ax.set_ylabel(r'$N_{sat}(r)$')
    for ax in axs[-1]:
        ax.set_xlabel(r'$r/r_{vir}$')
    plt.suptitle('Fit Results') 
    plt.savefig('results/fitresults.png', bbox_inches='tight')    

    fitres_txt = fitres_txt[:-3] # remove the last '\\ '
    stats_txt = stats_txt[:-3]
    # Textfile writing
    with open('results/fitresults.txt', 'w') as file:
        file.write(fitres_txt)
        file.close()
    with open('results/stats.txt', 'w') as file:
        file.write(stats_txt)
        file.close()
    with open('results/full_fitresults.txt', 'w') as file:
        file.write(full_fitres_txt_
        file.close()


def full_run():
    t0 = time.time()
    set_styles()
    maximization()
    make_plot_alldata()
    fit_data()
    print(f'Total Wall Runtime {time.time()-t0}s')
    

def main():
    full_run()


if __name__ == '__main__':
    main()


