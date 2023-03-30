import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matrix_class import Matrix
from ridder import ridders_method
from lu_decomp import lu_decomposition, solve_lineqs_lu

def func(x, a, b, c):
    return a/x + b*x + c

def compute_chi_sq(x, y, sigma, func, params):
    """Compute the chi squared value between N points x, y with
    y uncertainty sigma and a function func with parameters params"""
    return np.sum(((y - func(x, *params))**2)/(sigma*sigma))


def alpha_kl(dydp1, dydp2, sigma):
    """"""
    return np.sum((1./(sigma**2.)) * dydp1 * dydp2)

def beta_k(dchi_dp):
    """"""
    return -0.5 * dchi_dp

def levenberg_marquardt(xdata, ydata, sigma, func, guess, 
                        w=10, lmda=1e-3, chi_acc=0.1, max_iter=np.int(1e5), # fit procedure params
                        h_start=0.1, dec_factor=2, target_acc=1e-10): # derivative params
    """"""
    chi2 = compute_chi_sq(xdata, ydata, sigma, func, guess)

    N = len(xdata) # Number of data points
    M = len(guess) # Number of parameters
    A = Matrix(num_columns=M, num_rows=M)
    b = Matrix(num_columns=1, num_rows=M)
    params = guess

    for iteration in range(max_iter):
        # Get all derivatives of the function and chi squared wrt all parameters
        func_derivatives = np.zeros((M, N))
        chisq_derivatives = np.zeros(M)
        for i in range(M):
            # we want to pick out one of the parameters to diff over
            # should be a better way to do this right
            first_half_p = params[:i]
            if not i == M-1: # Avoid indexing errrors
                second_half_p = params[i+1:]            
            else:   
                second_half_p = []
            param_func = lambda p: [*first_half_p, p, *second_half_p]
            # Adjust Ridders method to do this in one go?
            for j in range(N):
                yp = lambda p: func(xdata[j], *param_func(p))
                dy_dpi = ridders_method(yp, [params[i]], h_start, dec_factor, target_acc)
                func_derivatives[i][j] = dy_dpi

            chi2_func_p = lambda p: compute_chi_sq(xdata, ydata, sigma, func, param_func(p))
            dchi_dpi = ridders_method(chi2_func_p, [params[i]], h_start, dec_factor, target_acc)
            chisq_derivatives[i] = dchi_dpi

        # Make A-matrix and b-vector
        for i in range(M):
            A.matrix[i][i] = (1.+lmda) * alpha_kl(func_derivatives[i], func_derivatives[i], sigma)
            b.matrix[i] = beta_k(chisq_derivatives[i])
            for j in range(i):
                A.matrix[i][j] = alpha_kl(func_derivatives[i], func_derivatives[j], sigma)
                A.matrix[j][i] = A.matrix[i][j]

        # Solve the set of linear equations for \delta p with LU decomposition
        LU = lu_decomposition(A)
        delta_p = solve_lineqs_lu(LU, b).matrix
        print(params)
        print(delta_p)
        # Evaluate new chi^2    
        # NOTE: We apply LU.row_order to the delta_p array, this might not be correct
        new_params = params + delta_p[LU.row_order].flatten() #
        new_chi2 = compute_chi_sq(xdata, ydata, sigma, func, new_params)
        delta_chi2 = new_chi2 - chi2
        print(lmda, delta_chi2)
        if delta_chi2 >= 0: # reject the solution
            lmda = w*lmda
            continue

        if np.abs(delta_chi2) < chi_acc:
            return params, new_chi2 # converged!
 
        # accept the step and make it
        params = new_params
        chi2 = new_chi2
        lmda = lmda/w        
    print("Max Iterations Reached"  )
    return params, new_chi2

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
    guess = [1., 1., 1.]
    fit_params, fit_chi2 = levenberg_marquardt(xpoints.flatten(), data.flatten(), sigma, func, guess, chi_acc=1e-5)
    popt, pcov = curve_fit(func, xpoints.flatten(), data.flatten(), p0=guess, sigma=np.repeat(sigma, num_x*N))
    scipy_chi2 = compute_chi_sq(xpoints, data, sigma, func, popt)
    for i in range(N):
        plt.scatter(xpoints[i], data[i], c='black', s=1)


    print(popt)
    print(fit_params)
    chi_real = compute_chi_sq(xpoints, data, sigma, func, params)
    plt.plot(x_fit, f(x_fit), c='red', label=rf'Analytic: chi_sq = {chi_real:.2E}')
    plt.plot(x_fit, func(x_fit, *fit_params), c='blue', label=f'Own Fit: chi_sq = {fit_chi2:.2E}')
    plt.plot(x_fit, func(x_fit, *popt), c='green', ls='--', label=f'SciPy: chi_sq = {scipy_chi2:.2E}')


    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.legend()
    plt.show()




def main():
    test_opti()

if __name__ == '__main__':
    main()
