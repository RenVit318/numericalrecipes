import numpy as np

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

    for ar_idx in range(len(x_ar)):
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
