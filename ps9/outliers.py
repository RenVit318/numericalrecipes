import numpy as np 
import matplotlib.pyplot as plt
from functions.fitting import levenberg_marquardt
from functions.minimization import quasi_newton, downhill_simplex

def L1(xdata, ydata, sigma, func, params):
    """Returns the negative log likelihood of an L1 estimator"""
    z = (ydata - func(xdata, *params))/sigma
    return np.sum(np.abs(z))

def lorentzian(xdata, ydata, sigma, func, params):
    """Returns the negative log likelihood of a Lorentzian estimator"""
    z = (ydata - func(xdata, *params))/sigma
    return np.sum(np.log(1. + 0.5*z*z))

def outlier_fitting():
    set_idx = 5
    data = np.genfromtxt(f'data/outliers_dataset{set_idx}.txt')
    # linear equation
    func = lambda x, a, b: a*x + b
    func_L1 = lambda x: L1(data[:,0], data[:,1], 1, func, x)
    func_lorentzian = lambda x: lorentzian(data[:,0], data[:,1], 1, func, x)

    # Fitting levenberg_marquardt with sigma = 1 is the same as least squares
    guess = np.array([20, 300], dtype=np.float64)
    fit_params, ls_fit, num_iter = levenberg_marquardt(data[:,0], data[:,1], 1, func, guess)
  
    # Now use an L1 optimizer to supress the effect of outliers
    mini_method = quasi_newton # DOWNHILL SIMPLEX NEEDS WORK TO FIT IN WITH FUNCTIONS
    L1_fit_params, _ = mini_method(func_L1, guess)
    lorentzian_fit_params, _ = mini_method(func_lorentzian, guess)
    print(fit_params)
    print(L1_fit_params)
    print(lorentzian_fit_params)

    xx = np.linspace(0, 100, 200)
    y_ls = func(xx, *fit_params) 
    y_l1 = func(xx, *L1_fit_params)
    y_lorentzian = func(xx, *lorentzian_fit_params)

    plt.scatter(data[:,0], data[:,1])
    plt.plot(xx, y_ls, c='black', ls='--', label=f'Standard Fit')# (LS={ls_fit})')
    plt.plot(xx, y_l1, c='red', ls='--', label=f'L1 Fit')
    plt.plot(xx, y_lorentzian, c='green', ls='--', label='Lorentizan Fit')
    plt.legend()
    plt.show()
    

def visualize_data():
    # 
    for i in range(1,6):
        data = np.genfromtxt(f'data/outliers_dataset{i}.txt')
        plt.scatter(data[:,0], data[:,1])
        plt.show()

def main():
    #visualize_data()
    outlier_fitting()

if __name__ == '__main__':
    main()
