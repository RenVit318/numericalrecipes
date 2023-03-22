import numpy as np

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


