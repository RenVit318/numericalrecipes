import numpy as np
import matplotlib.pyplot as plt
from matrix_class import Matrix
from ridder import ridders_method

def func(x, a, b, c):
    return a/x + b*x + c

def compute_chi_sq(x, y, sigma, func, params):
    """Compute the chi squared value between N points x, y with
    y uncertainty sigma and a function func with parameters params"""
    return np.sum(((y - func(x, *params))**2)/(sigma*sigma))


def alpha_kl():
    pass

def beta_k():
    pass


def levenberg_marquardt(xdata, ydata, sigma, func, guess, 
                        w=10, lmda=1e-3,  # fit procedure params
                        h_start=0.1, dec_factor=2, target_acc=1e-10): # derivative params
    """"""
    chi2 = compute_chi_sq(xdata, ydata, sigma, func, guess)
    converged = False
    N = len(xdata) # Number of data points
    M = len(guess) # Number of parameters
    A = Matrix(num_columns=M, num_rows=M)
    b = Matrix(num_columns=1, num_rows=M)
    params = guess

    while not converged:
        # Get all derivatives of the function and chi squared wrt all parameters and of
        func_derivatives = np.zeros((M, N))
        chisq_derivatives = np.zeros((M, N))
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

                chi2_func_p = lambda p: compute_chi_sq(xdata[j], ydata, sigma, func, param_func(p))
                dchi_dpi = ridders_method(chi2_func_p, [params[i]], h_start, dec_factor, target_acc)
                chisq_derivatives[i][j] = dchi_dpi

        # Make A-matrix
        for i in range(M):
            for j in range(i):
                print(i, j)

        converged = True
        
        # Make b-vector


def test_opti():
    params = [2, 1, 2]
    num_x = 20
    N = 10
    sigma = 2
    f = lambda x: func(x, *params)
    x = np.linspace(0.5, 4, num_x)
    x_fit = np.linspace(0.5, 5, 1000)

    xpoints = np.reshape(np.tile(x, N), (N, num_x))
    noise = np.random.normal(0, sigma, (N, num_x))
    data = f(xpoints) + noise

    # Optimization
    guess = [1., 1., 1.]
    levenberg_marquardt(xpoints.flatten(), data.flatten(), sigma, func, guess)

    for i in range(N):
        plt.scatter(xpoints[i], data[i], c='black', s=1)

    chi_real = compute_chi_sq(xpoints, data, sigma, func, params)
    plt.plot(x_fit, f(x_fit), c='red', label=rf'chi_sq = {chi_real:.2E}')
    
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.legend()
    plt.show()




def main():
    test_opti()

if __name__ == '__main__':
    main()
