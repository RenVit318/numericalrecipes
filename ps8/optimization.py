import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
from matrix_class import Matrix
from ridder import ridders_method
from lu_decomp import lu_decomposition, solve_lineqs_lu

def func(x, a, b, c):
    return a/x + b*x + c

def compute_chi_sq(x, y, sigma, func, params):
    """Compute the chi squared value between N points x, y with
    y uncertainty sigma and a function func with parameters params"""
    return np.sum(((y - func(x, *params))**2)/(sigma*sigma))

def make_param_func(params, i):
    """Given a list of parameters and an index i, return a function with
    p_i as the variable for use in differentation algorithms"""
    # should be a better way to do this right
    first_half_p = params[:i]
    if not i == len(params)-1: # Avoid indexing errrors
        second_half_p = params[i+1:]            
    else:   
        second_half_p = []
    return lambda p: [*first_half_p, p, *second_half_p]


def make_alpha_matrix(xdata, sigma, func, params,
                      h_start=0.1, dec_factor=2, target_acc=1e-10): #derivative params
    """Make a Matrix object containing the sum of N products of derivatives
    where the element i,j is the product of df/dxi and df/dxj. Each value i 
    can be weighted by its uncertainty sigma if desired. If this is not 
    required one can set sigma = 1 to 'ignore' this step"""
    N = len(xdata) # Number of data points
    M = len(params) # Number of parameters
    A = Matrix(num_columns=M, num_rows=M)

    func_derivatives = np.zeros((M, N))

    # Build up all M derivatives
    for i in range(M):
        param_func = make_param_func(params, i)
        # Adjust Ridders method to do this in one go? Big speed upgrade.
        for j in range(N):
            yp = lambda p: func(xdata[j], *param_func(p))
            dy_dpi = ridders_method(yp, [params[i]], h_start, dec_factor, target_acc)
            func_derivatives[i][j] = dy_dpi

    # Build up A-matrix
    for i in range(M):
        A.matrix[i][i] = alpha_kl(func_derivatives[i], func_derivatives[i], sigma)
        for j in range(i):
            A.matrix[i][j] = alpha_kl(func_derivatives[i], func_derivatives[j], sigma)
            A.matrix[j][i] = A.matrix[i][j]

    return A

def make_nabla_chi2(xdata, ydata, sigma, func, params,
                    h_start=0.1, dec_factor=2, target_acc=1e-10):
    """"""
    M = len(params)
    chisq_derivatives = np.zeros(M)

    for i in range(M):
        param_func = make_param_func(params, i)
        chi2_func_p = lambda p: compute_chi_sq(xdata, ydata, sigma, func, param_func(p))
        dchi_dpi = ridders_method(chi2_func_p, [params[i]], h_start, dec_factor, target_acc)
        chisq_derivatives[i] = dchi_dpi

    return beta_k(chisq_derivatives)


def weigh_A_diagonals(A, lmda):
    """"Weigh the diagonal elements of a square matrix A by a factor (1+lmda)""" 
    if not A.matrix.shape[0] == A.matrix.shape[1]:
        raise ValueError(f"This Matrix object is not square: {A.matrix}")
    for i in range(A.matrix.shape[0]):
        A.matrix[i][i] *= (1. + lmda)
    return A


def alpha_kl(dydp1, dydp2, sigma):
    """"""
    return np.sum((1./(sigma**2.)) * dydp1 * dydp2)

def beta_k(dchi_dp):
    """"""
    return -0.5 * dchi_dp

def levenberg_marquardt(xdata, ydata, sigma, func, guess,
                        w=10, lmda=1e-3, chi_acc=0.1, max_iter=int(1e5), # fit procedure params
                        h_start=0.1, dec_factor=2, target_acc=1e-10): # derivative params
    """"""
    chi2 = compute_chi_sq(xdata, ydata, sigma, func, guess)

    N = len(xdata) # Number of data points
    M = len(guess) # Number of parameters
    A = Matrix(num_columns=M, num_rows=M)
    b = Matrix(num_columns=1, num_rows=M)
    params = guess

    # Can do this beforehand because the derivatives never change
    # if the functions depend linearly on the parameters
    A = make_alpha_matrix(xdata, sigma, func, params, h_start, dec_factor, target_acc)

    for iteration in range(max_iter):
        A_weighted = copy.deepcopy(A) # ensure no pointing goes towards A
        A_weighted = weigh_A_diagonals(A_weighted, lmda) # Make \alpha_prime
        
        b.matrix = make_nabla_chi2(xdata, ydata, sigma, func, params, h_start, dec_factor, target_acc)
          
        # Solve the set of linear equations for \delta p with LU decomposition
        LU = lu_decomposition(A_weighted, implicit_pivoting=True)
        delta_p = solve_lineqs_lu(LU, b).matrix
   
        # Evaluate new chi^2    
        new_params = params + delta_p.flatten()
        new_chi2 = compute_chi_sq(xdata, ydata, sigma, func, new_params)
        delta_chi2 = new_chi2 - chi2
        if delta_chi2 >= 0: # reject the solution
            lmda = w*lmda
            continue

        if np.abs(delta_chi2) < chi_acc:
            return params, new_chi2, iteration+1 # converged!
 
        # accept the step and make it
        params = new_params
        chi2 = new_chi2
        lmda = lmda/w        
    print("Max Iterations Reached")
    return params, new_chi2, iteration+1

def test_opti():
    params = [2, 1, 2]
    num_x = 20
    N = 1000
    sigma = 1
    f = lambda x: func(x, *params)
    x = np.linspace(0.5, 4, num_x)
    x_fit = np.linspace(0.5, 5, 1000)

    xpoints = np.reshape(np.tile(x, N), (N, num_x))
    noise = np.random.normal(0, sigma, (N, num_x))
    data = f(xpoints) + noise

    # Optimization
    guess = [0.,0.,0.]
    fit_params, fit_chi2, num_iter = levenberg_marquardt(xpoints.flatten(), data.flatten(), sigma, func, guess, chi_acc=1e-10)
    print('Own fit done! Num Iterations: ', num_iter)
    popt, pcov = curve_fit(func, xpoints.flatten(), data.flatten(), p0=guess, sigma=np.repeat(sigma, num_x*N))
    scipy_chi2 = compute_chi_sq(xpoints, data, sigma, func, popt)
    for i in range(N):
        plt.scatter(xpoints[i], data[i], c='black', s=1)


    print(popt)
    print(fit_params)
    chi_real = compute_chi_sq(xpoints, data, sigma, func, params)
    plt.plot(x_fit, f(x_fit), c='red', label=rf'Analytic: chi_sq = {chi_real:.2E}')
    plt.plot(x_fit, func(x_fit, *fit_params), c='blue', label=f'Own Fit: chi_sq = {fit_chi2:.2E}')
    #plt.plot(x_fit, func(x_fit, *popt), c='green', ls='--', label=f'SciPy: chi_sq = {scipy_chi2:.2E}')


    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.legend()
    plt.show()




def main():
    test_opti()

if __name__ == '__main__':
    main()
