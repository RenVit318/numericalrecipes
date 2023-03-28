############
#
# This file contains almost all extra functions develloped for the problem classes
# and now to be used for the exercises of hand in 2
#
#############

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
# TODO: Improve this into a better method 
def rejection_sampling(func, rng, N, 
                       shift_x=lambda x: x,
                       shift_y=lambda x: x,
                       x0 = 4891653):
    """Sample a distribution using rejection sampling
    Expand documentation!"""
    
    sampled_points = np.zeros(N)
    num_tries = 0 # For performance testing
    for i in range(N):
        not_sampled = True

        # Keep sampling until we find a x,y pair that fits
        while not_sampled:
            numbers, x0 = rng(2, x0=x0, return_laststate=True) # This is now U(0,1)    
             
            x = shift_x(numbers[0])
            y = shift_y(numbers[1])
            num_tries += 1
            print(x, y, func(x) )
            if y < func(x):
                sampled_points[i] = x
                not_sampled = False

        if (i%100) == 0:
            print(i)

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


# SORTING
def sort_subarrays(a1, a2, k1, k2):
    """Takes two subarrays 1,2 with indices a1, a2 and values k1, k2 and combines these
    into array with indices a such that the k are sorted in ascending order"""
    N1 = len(a1)  
    N2 = len(a2)

    # We built up a new instance of our sorting array. This is not memory efficient
    # TODO: Study the algorithm below to try and find a better method
    a_sorted = np.zeros(N1+N2)
    
    if N1 == 0: 
        return a2
    if N2 == 0:
        return a1
    
    # Walk through the left- and right- sub arrays separately
    idx1 = 0
    idx2 = 0
    while True:
        if k1[idx1] > k2[idx2]: 
            # Then place the second element to the left of the first element and take 
            # one step to the right in the right sub-array
            a_sorted[idx1+idx2] = a2[idx2]
            idx2 += 1

            # Then, if there are elements remaining in the right array, keep
            # placing them to the left as long as they're smaller
            if idx2 < N2: # need this if statement to save us from indexing errors in the while
                while (k1[idx1] > k2[idx2]):
                    a_sorted[idx1+idx2] = a2[idx2]  
                    idx2 += 1

                    if idx2 >= (N2):
                        # No more elements left in the right array, we can fill out with the left array
                        for j in range(idx1, N1):
                            a_sorted[j+idx2] = a1[j]
                        return a_sorted
                       
                # Now the element from the left array is smaller than the first remaining element
                # from the right array, we can safely place it
                a_sorted[idx1+idx2] = a1[idx1]
                idx1 += 1   
            else: 
                # No more elements left in the right sub-array
                for j in range(idx1, N1):
                    a_sorted[j+idx2] = a1[j]
                return a_sorted

        else:  
            a_sorted[idx1+idx2] = a1[idx1]
            idx1 += 1

        # Check if we have reached the end of the left sub-array
        # If we have, fill out the rest of the array with the right sub-array
        if idx1 == N1:
            for j in range(idx2, N2):
                a_sorted[idx1+j] = a2[j]
            return a_sorted



def merge_sort(a=None, key=None):
    """Sorts the array or list using merge sort. This function iteratively
    builds up the array from single elements which are sorted by sort_subarrays
    Note, in principle one should only provide either 'a' or 'key':

    - If 'a' is provided, that array is sorted in ascending order, and returned 
      by this function
    - If 'key' is provided, this function returns the indices corresponding to the
      order in which the array would be sorted
    - If both 'a' and 'key' are provided, this function assumes 'key' has already
      been previously shuffled, and just swaps indices of the preexisting 'a'
   
    RETURNS: 'a': numpy array
    """
    if key is not None:
        key = np.array(key)

    if a is None and key is not None:
        a = np.arange(len(key))  

    a = np.array(a) 
    subsize = 1    
    N = len(a)
    is_sorted = False
    # Build up the array sorting arrays of increasing subsize
    while not is_sorted:
        subsize *= 2
        if subsize > N:
            is_sorted = True # After this iteration, the array is sorted

        for i in range(int(np.ceil(N/subsize))):
            # We need the min(.. , N) to ensure that we do not exceed the length of the 
            # array with our indexing
            subarray1 = a[i*subsize: i*subsize+int(0.5*subsize)] # First half of the interval
            subarray2 = a[i*subsize+int(0.5*subsize): np.min(((i+1)*subsize, N))]
            if key is not None:
                key1, key2 = key[subarray1] , key[subarray2]
                sorted_sub = sort_subarrays(subarray1, subarray2, key1, key2)
            else:
                # we feed in 'subarrayx' twice because if we only sort a, a is its own key
                sorted_sub = sort_subarrays(subarray1, subarray2, subarray1, subarray2)

            a[i*subsize:subsize*(i+1)] = sorted_sub
           
    return a


# ROOT FINDING
def false_position(func, bracket, target_x_acc=1e-3, target_y_acc=1e-8, max_iter=int(1e5)):
    """Given a function f(x) and a bracket [a,b], this function returns
    a value c within that interval for which f(c) ~= 0 using the false
    position method. Guaranteed to converge, slow but faster than bisection.

    Inputs:

    Outputs:
    """
    a, b = bracket
    fa, fb = func(a), func(b) 

    # Test if the bracket is good
    if not (fa*fb) < 0:
        raise ValueError("The provided bracket does not contain a root")

    for i in range(max_iter):
        if np.abs(a-b) < target_x_acc:
            # We made our x-axis precision. Select wichever one is closer to 0
            best_root = [a,b][np.argmin(np.abs([fa,fb]))]
            return best_root, i+1

        c = b + fb * ((b-a)/(fa-fb))
        fc = func(c)

        if np.abs((fc-fb)/fc) < target_y_acc or np.abs((fc-fa)/fc) < target_y_acc:
            return c, i+1
        
        if (fa*fc) < 0:
            # Then the new interval becomes a, c
            # keep the number of function calls as low as possible
            b, fb = c, fc
        elif (fc*fb) < 0:
            # Then the new interval becomes c, b
            a, fa = c, fc
        else:
            print("Warning! Bracket seems to have diverged")
            return c, i+1

        

    print("Maximum number of iterations reached")
    return c, i
 
