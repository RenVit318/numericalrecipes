#############
#
# Scripts for Q2 on Hand-in Assignment 1
#
#############

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matrix_class import Matrix
from matrix_functions import lu_decomposition, check_solution, solve_lineqs_lu
from interpolation import poly_interpolator
import timeit
from plotting import set_styles


def import_data():
    """Import the Vandermonde data and place it in a Vandermonde matrix"""
    data = np.genfromtxt(os.path.join(sys.path[0], "Vandermonde.txt"), comments='#', dtype=np.float64)
    x = data[:, 0]
    y = data[:, 1]
    x_interp = np.linspace(x[0], x[-1], 1001)  # x values to interpolate at

    V = np.zeros((len(x), len(x)))
    # Fill out the Vandermonde Matrix as V_ij = x_i^j
    for j in range(len(x)):
        V[:, j] = x**j

    return V, x, y, x_interp


def compute_polynomial(x, coeff):
    """Computes a polynomial evaluated at all x, with coefficients coeff"""
    y = np.zeros_like(x)
    for i in range(len(coeff)):
        y += coeff[i] * x**i
    return y


def LU_decomposition(V, x, y, x_interp, num_iterations=0):
    """Performs LU decomposition on the Vandermonde matrix V to find the matrices L and U such that L*U = V
    It then uses these decomposed matrices to solve for the coefficient of the polynomial that goes through
    all points x_i, y_i. If num_iterations > 0, we reapply the LU matrix to \delta y = Vc' - y. """
    LU = lu_decomposition(V)
    LU_coefficients = solve_lineqs_lu(LU, y)
    

    # Evaluate the Vandermonde polynomial on the whole smooth range, and at the 20 data points
    LU_polynomial = compute_polynomial(x_interp, LU_coefficients.matrix)
    LU_poly_at_xdata = compute_polynomial(x, LU_coefficients.matrix)

    # Iterative Improvement
    for i in range(num_iterations):
        delta_y = LU_poly_at_xdata - y
        delta_coefficients = solve_lineqs_lu(LU, delta_y)
        LU_coefficients.matrix -= delta_coefficients.matrix

        LU_polynomial = compute_polynomial(x_interp, LU_coefficients.matrix)
        LU_poly_at_xdata = compute_polynomial(x, LU_coefficients.matrix)

    return LU_polynomial, LU_poly_at_xdata


def neville_fit(x, y, x_interp):
    """Uses Neville's Algorithm to fit a polynomial of order M=len(data)-1 to the data points"""
    neville_y, neville_unc = poly_interpolator(x, y, x_interp, len(x))
    neville_y_at_x, delta_neville_unc = poly_interpolator(x, y, x, len(x))
    return neville_y, neville_y_at_x


def vandermonde_fit(num_LU_iterations=10):
    """Code for Assignment 2 from Handout 1"""
    V, x, y, x_interp = import_data()

    # a. LU Decomposition
    LU_y, LU_y_at_x = LU_decomposition(V, x, y, x_interp)

    # b. Neville's Algorithm
    neville_y, neville_y_at_x = neville_fit(x, y, x_interp)

    # c. Iterative LU improvement
    LU_y_iterative, LU_y_at_x_iterative = LU_decomposition(V, x, y, x_interp, num_LU_iterations)

    # Plot Data
    set_styles()
    fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(10,8))
    ax0.plot(x, y, marker='o', linewidth=0)
    ax0.plot(x_interp, LU_y, c='C1', label='LU Decomposition')
    ax0.plot(x_interp, neville_y, c='C2', ls='dashed', label="Neville's Algorithm")
    ax0.plot(x_interp, LU_y_iterative, c='C3', ls='dotted', label=f'{num_LU_iterations} Times Iterated LU')
    ax0.legend()

    ax1.plot(x, np.abs(y - LU_y_at_x), c='C1', marker='o', lw=0)
    ax1.plot(x, np.abs(y - neville_y_at_x), c='C2', marker='o', lw=0)
    ax1.plot(x, np.abs(y - LU_y_at_x_iterative), c='C3', marker='o', lw=0)
    ax1.axhline(y=0, c='black', ls='--', alpha=0.6)
    ax1.set_yscale('log')

    plt.xlim(-1, 101)
    ax0.set_ylim(-400, 400)
    ax1.set_xlabel('$x$')
    ax0.set_ylabel('$y$')
    ax1.set_ylabel(r'Absolute $\Delta y$')

    plt.suptitle("Lagrange Polynomial Estimations"          )
    plt.savefig('results/vandermonde_fitresults.png', bbox_inches='tight')
    #plt.show()


def vandermonde_timeit(num_iter=100):
    """Time the execution time of the code snippets above"""
    V, x, y, x_interp = import_data()

    time_LU = timeit.timeit(f"LU_decomposition(V, x, y, x_interp, num_iterations=0)",
                            setup='from __main__ import LU_decomposition, import_data \
                                   \nV, x, y, x_interp = import_data()',
                            number=num_iter)/num_iter

    time_LU_iterative = timeit.timeit("LU_decomposition(V, x, y, x_interp, num_iterations=10)",
                                      setup='from __main__ import LU_decomposition, import_data \
                                             \nV, x, y, x_interp = import_data()',
                            number=num_iter)/num_iter

    time_neville = timeit.timeit("neville_fit(x, y, x_interp)",
                                  setup='from __main__ import neville_fit, import_data \
                                         \nV, x, y, x_interp = import_data()',
                                  number=num_iter)/num_iter

    table = f"""Algorithm & Runtime (ms) \\\\
\hline
LU Decomposition (1x) & {time_LU*1e3:.2f} \\\\
LU Decomposition (10x) & {time_LU_iterative*1e3:.2f} \\\\
Neville's Algorithm & {time_neville*1e3:.2f}"""

    with open('results/vandermonde_timetab.txt', 'w') as f:
        f.write(table)

    #print('almost done.')
    #input()

def main():
    vandermonde_fit()
    print("\tTiming Algorithms..")
    vandermonde_timeit()


if __name__ == '__main__':
    main()
