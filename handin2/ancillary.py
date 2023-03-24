import numpy as np


# INTEGRATION
def romberg_integration(func, a, b, order, open_formula=False):
    """Integrate a function using Romberg Integration"""
    # initiate all parameters
    r_array = np.zeros(order)
    h = b - a
    N_p = 1

    # fill in first estimate, don't do this if we cant evaluate at the edges
    if open_formula:
        # First estimate will be with h = (b-a)/2
        start_point = 0
    else:
        r_array[0] = 0.5*h*(func(b) - func(a))
        start_point = 1

    
    # First iterations to fill out estimates of order m
    for i in range(start_point, order):
        delta = h
        h *= 0.5
        x = a + h

        # Evaluate function at Np points
        for j in range(N_p):
            r_array[i] += func(x)
            x += delta
        # Combine new function evaluations with previous 
        r_array[i] = 0.5*(r_array[i-1] + delta*r_array[i])
        N_p *= 2
    
    # Combine all of our estimations to cancel our error terms
    N_p = 1
    for i in range(1,order):
        N_p *= 4
        for j in range(order-i):
            r_array[j] = (N_p*r_array[j+1] - r_array[j])/(N_p-1)

    return r_array[0]

# DISTRIBUTION SAMPLING
def rejection_sampling(func, rng, N, 
                       shift_x=lambda x: x,
                       shift_y=lambda x: x,
                       x0 = 4891653):
    """Sample a distribution using rejection sampling
    Expand documentation!"""
    
    sampled_points = np.zeros(N)
    num_tries = 0 # For testing purposes
    for i in range(N):
        not_sampled = True

        # Keep sampling until we find a x,y pair that fits
        while not_sampled:
            numbers, x0 = rng(2, x0=x0, return_laststate=True) # This is now U(0,1)    
             
            x = shift_x(numbers[0])
            y = shift_y(numbers[1])
            num_tries += 1
            if y < func(x):
                sampled_points[i] = x
                not_sampled = False

    print(f'Average No. tries: {num_tries/N:.1f}')
    return sampled_points

# DIFFERENTIATION
def central_difference(func, x, h):
    """Comptue the derivative of a function evaluated at x, with step size h"""
    return (func(x+h) - func(x-h))/(2.*h)   

def ridders_equation(D1, D2, j, dec_factor):
    j_factor = dec_factor**(2.*(j+1.))
    return (j_factor * D2 - D1)/(j_factor - 1)
    

def ridders_method(func, x_ar, h_start, dec_factor, target_acc, approx_array_length=15):
    """Compute the derivative of a function at a point, or points x using Ridder's Method

    Inputs:
        func:
        x_ar
        h_start
        dec_factor
        target_acc
        approx_array_length

    Outputs:
        

    """
    derivative_array = np.zeros_like(x_ar, dtype=np.float64)
    unc_array = np.zeros_like(x_ar, dtype=np.float64)

    for ar_idx in range(len(x_ar)):
        x = x_ar[ar_idx]
        # Make this larger if we have not reached our target accuracy yet
        approximations = np.zeros(approx_array_length, dtype=np.float64)
        uncertainties = np.zeros(approx_array_length, dtype=np.float64)
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

    return derivative_array, unc_array  
