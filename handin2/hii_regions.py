import numpy as np
import matplotlib.pyplot as plt
import time
from ancillary import FP_NR
from plotting import set_styles
import timeit


def equilibrium1(T, Z, Tc, psi, k):
    """Simple equilibrium temperature equation, copied from lecture files"""
    return psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T


def equilibrium2(T, Z, Tc, psi, nH, A, xi, k, aB):
    """More realistic equilibrium temperature equation, copied from lecture files"""
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)


def full_run():
    ## Variables
    aB = 2e-13 # cm^3 s^-1
    k= 1.38e-16 # erg/K
    psi = 0.929 #
    Tc = 1e4 # K
    Z = 0.015 #
    A = 5e-10 # erg
    xi = 1e-15 # s^-1

    # Bracketing Params.
    Tmin = 1
    Tmax1 = 1e7
    Tmax2 = 1e15
    rootfinder = FP_NR
    ####
    set_styles()

    # 2a
    equilibrium1_func = lambda x: equilibrium1(x, Z, Tc, psi, k)
    t0 = time.time()
    root, num_iter = rootfinder(equilibrium1_func, [Tmin, Tmax1], target_x_acc=0.1, target_y_acc=1e-15)
    t1 = time.time()
    y_root = equilibrium1_func(root)
    print(f'Root found for equation 1 using FP_NR:\nT={root:.2E}K (f(T)={y_root:.8E}\nNum Iter: {num_iter}, time:{t1-t0:.2E}s\n')

    # Plot the function
    x = np.linspace(Tmin, Tmax1, 1000)
    y = equilibrium1_func(x)
    plt.plot(x, y, label='Equilibrium Function')
    plt.axhline(y=0, c='black', ls='--', alpha=0.8)

    plt.scatter([Tmin, Tmax1], [equilibrium1_func(Tmin), equilibrium1_func(Tmax1)], c='red', label='Starting Points', marker='X')
    plt.scatter(root, y_root, c='green', label=f'Root T={root:.2E}K (f(T)={y_root:.2E})', marker='X')

    plt.xlabel(r'$^{10}\log~T$')
    plt.ylabel(r'$\propto\,\Gamma_{pe}~-~\Lambda_{rr}$')
    plt.title('Simple HII Region Equilibrium')

    plt.xlim(Tmin, Tmax1)
    plt.xscale('log')
    plt.legend()
    plt.savefig('results/simple_hiiregion_roots.png', bbox_inches='tight')
    plt.clf()

    #2b
    table_txt = ""
    for i, n_power in enumerate([-4, 0, 4]):
        n = 10**(n_power)
        equilibrium2_func = lambda x: equilibrium2(x, Z, Tc, psi, n, A, xi, k, aB)
        t0 = time.time()
        root, num_iter = rootfinder(equilibrium2_func, [Tmin, Tmax2], target_y_acc=1e-10, target_x_acc=1e-10)
        t1 = time.time()
        y_root = equilibrium2_func(root)

        x = np.logspace(np.log10(Tmin), np.log10(Tmax2), 5000)
        y = equilibrium2_func(x)

        plt.plot(x, np.abs(y), c=f'C{i}',   label=rf'$n$ = {n:.0E} cm$^{-3}$')
        plt.scatter(root, np.abs(y_root), label=f'T={root:.2E}', c=f'C{i}', marker='X', zorder=0)
        table_txt += rf"$10^{{{n_power}}}$ & {root:.2E} & {y_root:.2E} & {num_iter} & {t1-t0:.2}\\" +'\n'
        print(f'Root found for equation 2, n=10^{n_power} using FP_NR:\nT={root:.2E}K (f(T)={y_root:.8E}\nNum Iter: {num_iter}, time:{t1-t0:.2E}s\n')
    plt.xlabel(r'$^{10}\log~T$')
    plt.ylabel(r'$\propto\,^{10}\log|\Gamma~-~\Lambda$|')
    plt.title('Complex HII Region Equilibrium')

    plt.xlim(Tmin, Tmax2)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('results/complex_hiiregion_roots.png', bbox_inches='tight')

    table_txt = table_txt[:-4] # remove the last \\\n
    with open('results/complex_hiiregion_table.txt', 'w') as file:
        file.write(table_txt)



def main():
    full_run()

if __name__ == '__main__':
    main()
