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
from matrix_functions import lu_decomposition, check_lu_decomposition, check_solution, solve_lineqs_lu
from interpolation import poly_interpolator

def import_data():
    """Import the Vandermonde data and place it in a Vandermonde matrix"""
    data = np.genfromtxt(os.path.join(sys.path[0], "vandermonde_data.txt"), comments='#', dtype=np.float64)
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



def vandermonde_fit(num_LU_iterations=10):
    """Code for Assignment 2 from Handout 1"""
    V, x, y, x_interp = import_data()

    # a. LU Decomposition
    LU = lu_decomposition(V)
    #_ = check_lu_decomposition(LU, V)
    LU_coeff = solve_lineqs_lu(LU, y)
    LU_y = compute_polynomial(x_interp, LU_coeff.matrix)
    LU_y_at_x = compute_polynomial(x, LU_coeff.matrix)

    # b. Neville's Algorithm
    neville_y, neville_unc = poly_interpolator(x, y, x_interp, len(x))
    neville_y_at_x, delta_neville_unc = poly_interpolator(x, y, x, len(x))

    # c. Iterative LU improvement
    LU_coeff_iterative = LU_coeff
    LU_y_at_x_iterative = LU_y_at_x
    delta_b = LU_y_at_x - y

    for i in range(num_LU_iterations):
        improvement = solve_lineqs_lu(LU, delta_b)
        LU_coeff_iterative.matrix += improvement.matrix
        LU_y_at_x_iterative = compute_polynomial(x, LU_coeff_iterative.matrix)
        delta_b = LU_y_at_x_iterative - y

    LU_y_iter = compute_polynomial(x_interp, LU_coeff_iterative.matrix)



    # Plot Data
    fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(10,8))
    ax0.plot(x, y, marker='o', linewidth=0)
    ax0.plot(x_interp, LU_y, c='C1', label='LU Decomposition')
    ax0.plot(x_interp, neville_y, c='C2', label="Neville's Algorithm")
    ax0.plot(x_interp, LU_y_iter, c='C3', label=f'{num_LU_iterations} Times Iterated LU')
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

    plt.suptitle("Vandermonde Matrix Fit Results")

    plt.show()


def main():
    vandermonde_fit()
    #import_data()

if __name__ == '__main__':
    main()
