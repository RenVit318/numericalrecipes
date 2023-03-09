import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return (x**2.)*np.sin(x)

def df_dx_analytical(x):
    return 2.*x*np.sin(x) + (x**2.) * np.cos(x)

def central_difference(func, x, h):
    """Comptue the derivative of a function evaluated at x, with step size h"""
    return (func(x+h) - func(x-h))/(2.*h)   

def ridders_equation(D1, D2, j, dec_factor):
    j_factor = dec_factor**(2.*(j+1.))
    return (j_factor * D2 - D1)/(j_factor - 1)
    

def ridders_method(func, x_ar, h_start, dec_factor, target_acc, approx_array_length=15):
    """"""
    derivative_array = np.zeros_like(x_ar)
    unc_array = np.zeros_like(x_ar)

    for ar_idx in range(x_ar.shape[0]):
        x = x_ar[ar_idx]

        # Make this larger if we have not reached our target accuracy yet
        approximations = np.zeros(approx_array_length)
        uncertainties = np.zeros(approx_array_length)
        uncertainties[0] = np.inf # set uncertainty arbitrarily large for the error improvement comparison

        h_i = h_start
        approximations[0] = central_difference(func, x, h_i)
        best_guess = approximations[0]

        for i in range(1, approx_array_length):            
            # Add in a new estimation with smaller step size
            h_i /= dec_factor
            approximations[i] = central_difference(func, x, h_i)
            for j in range(i):
                # Add the new approximation into the 'tree of estimations'
                approximations[i-j-1] = ridders_equation(approximations[i-j-1], approximations[i-j], j, dec_factor) 
            uncertainties[i] = np.abs(approximations[0] - best_guess)

            # Test if we are below our target accuracy
            if (uncertainties[i] < target_acc) or (uncertainties[i] > uncertainties[i-1]):
                derivative_array[ar_idx] = approximations[0]
                unc_array[ar_idx] = uncertainties[i]
                break
            else:
                best_guess = approximations[0]

    return derivative_array     
        

def diff_func():
    h_vals = [0.1, 0.01, 0.001]
    ridder_h_start = 1
    ridder_d = 2
    target_acc = 1e-2
    x = np.linspace(0, 2*np.pi, 101)

    fig, (ax0, ax1) = plt.subplots(2,1)
#    ax0.plot(x, func(x), label='f(x)', c='black')
    ax0.plot(x, df_dx_analytical(x), label='df(x)/dx Analytical', c='black')

    #for h in h_vals:
    #    plt.plot(x, central_difference(func, x, h), label=f'df(x)/dx CD h={h}', ls='--')

    for target_acc in [1e-10]:#[1e-2, 1e-5, 1e-8, 1e-10, 1e-15]:
        ridder_derivative = ridders_method(func, x, ridder_h_start, ridder_d, target_acc)
        ax0.plot(x, ridder_derivative, ls='--', label=f'Ridder Acc={target_acc}')
        
      
        ax1.plot(x, np.abs(ridder_derivative - df_dx_analytical(x)))
    ax0.legend()
    plt.yscale('log')
    plt.show()

def main():
    diff_func()
#    ridders_method(func, np.array([1]), 0.1, 2, 1e-5)


if __name__ == '__main__':
    main()
