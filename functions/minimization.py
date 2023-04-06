#################
#
# Functions for 1 and N-Dimensional minimization of functions
# Important functions:
#   - quasi_newton
#   - downhill_simplex
# 
#################

import numpy as np
from .algebra import ridders_method
from scipy.optimize import fmin  # REMOVE LATER

# CODE FOR 1-DIMENSIONAL FUNCTION MINIMIZATION
def parabola_min_analytic(a, b, c, fa, fb, fc):
    """Analytically computes the x-value of the minimum of a parabola
    that crosses a, b and c
    """
    top = (b-a)**2 * (fb-fc)  - (b-c)**2 * (fb-fa)
    bot = (b-a) * (fb-fc) - (b-c) * (fb-fa)
    return b - 0.5*(top/bot)


def make_bracket(func, bracket, w=(1.+np.sqrt(5))/2, dist_thresh=100, max_iter=10000):
    """Given two points [a, b], attempts to return a bracket triplet
    [a, b, c] such that f(a) > f(b) and f(c) > f(b).
    Note we only compute f(d) once for each point to save computing time"""
    a, b = bracket
    fa, fb = func(a), func(b)
    direction = 1 # Indicates if we're moving right or left
    if fa < fb:
        # Switch the two points
        a, b = b, a
        fa, fb = fb, fa
        direction = -1 # move to the left

    c = b + direction * (b - a) *w
    fc = func(c)
    
    for i in range(max_iter):
        if fc > fb:
            return np.array([a, b, c])  , i

        d = parabola_min_analytic(a, b, c, fa, fb, fc)
        fd = func(d)
        # We might have a bracket if b < d < c
        if (d>b) and (d<c):
            if fd > fb:
                return np.array([a, b, d]), i
            elif fd < fc:
                return np.array([b, d, c]), i
            # Else we don't want this d
            print('no parabola, in between b and c')
            d = c + direction * (c - b) * w
        elif (d-b) > 100*(c-b): # d too far away, don't trust it
            print('no parabola, too far away')
            d = c + direction * (c - b) * w
        elif d < b:
            print('d smaller than b')

        # we shifted but didn't find a bracket. Go again
        a, b, c = b, c, d
        fa, fb, fc = fb, fc, fd

    print('WARNING: Max. iterations exceeded. No bracket was found. Returning last values')
    return np.array([a, b, c]), i


def golden_section_search(func, bracket, target_acc=1e-5, max_iter=int(1e5)):
    """Once we have a start 3-point bracket surrounding a minima, this function iteratively
    tightens the bracket to search of the enclosed minima using golden section search."""
    w = 2. -  (1.+np.sqrt(5))/2 # 2 - golden ratio
    a, b, c = bracket
    fa, fb, fc = func(a), func(b), func(c)
    
    for i in range(max_iter):
        # Set new point in the largest interval
        # We do this separately because the bracket propagation can just not be generalized sadly
        if np.abs(c-b) > np.abs(b-a): # we tighten towards the right
            d = b + (c-b)*w
            fd = func(d)
            if fd < fb: # min is in between b and c
                a, b, c = b, d, c
                fa, fb, fc = fb, fd, fc
            else: # min is in between a and d
                a, b, c = a, b, d 
                fa, fb, fc = fa, fb, fd
        else: # we tighten towards the left
            d = b + (a-b)*w
            fd = func(d)
            if fd < fb: # min is in between a and b
                a, b, c = a, d, b
                fa, fb, fc = fa, fd, fb
            else: # min is in between d and c
                a, b, c = d, b, c
                fa, fb, fc = fd, fb, fc            
        
        if np.abs(c-a) < target_acc:
            return [b,d][np.argmin([fb, fd])], i+1 # return the x point corresponding to the lowest f(x)

    print("Maximum Number of Iterations Reached")
    return b, i+1


# CODE FOR THE N-DIMENSIONAL DOWNHILL SIMPLEX
def compute_centroid(A):
    """Compute the centroid of N points in N dimensional space. x should
    be an NxN array of N vectors with N dimensions (in that order). The function
    then returns one ndarray of length N with the centroid coordinates."""
    return (1./A.shape[1]) * np.sum(A, axis=0)


def downhill_simplex(func, start, shift_func=lambda x: x+1, max_iter=int(1e5), target_acc=1e-5):
    """Finds the minimum of a function using the downhill simplex method
    INPUT:
        func: A function taking only one variable as input with dimension N
        start: N-Dimensional numpy array where the function starts searching for a minimum
        shift_func: A function taking only one float as input dictating how to mutate the
                    initial simplex vertices

    OUTPUT:
        
    """
    dim = start.shape[0] # = N
    # Store N+1 vertice vectors in this matrix. This ordering fails if we feed it directly to func,
    # but it allows us to choose a vertex as vertices[i]. The function problem we solve by just   
    # transposing this this matrix.
    vertices = np.zeros((dim+1, dim)) 
    func_vals = np.zeros(dim+1) # Store the f(X) values in here

    # Create the simplex, add slight variation to each vector except the first using 'shift_func'
    vertices[0] = start
    for i in range(dim):
        vertices[i+1] = vertices[0]
        vertices[i+1][i] = shift_func(vertices[i+1][i])
    print(vertices)
    func_vals = func(vertices.T) 

    # Start algorithm
    for i in range(1, max_iter+1): 
        # Sort everything by function value
        sort_idxs = merge_sort(key=func_vals)
        vertices = vertices[sort_idxs]
        func_vals = func_vals[sort_idxs]

        # Check if we have reached our accuracy level by comparing the best and worst function evals.
        accuracy =(np.abs(func_vals[-1] - func_vals[0])/np.abs(0.5*(func_vals[-1] + func_vals[0]))) 

        if accuracy < target_acc:
            print(accuracy, target_acc)
            return vertices[0], i # corresponds to func_vals[0], so the best point

        # Compute the centroid of all but the last (worst) point
        centroid = compute_centroid(vertices[:-1])
        
        # Try out a new points
        x_try = 2.* centroid - vertices[-1]
        f_try = func(x_try)
        
        if f_try < func_vals[-1]:
            # There is improvement in this step
            if f_try < func_vals[0]:
                # We are the best point. Try expanding
                x_exp = 2*x_try - centroid
                f_exp = func(x_exp)
                if f_exp < f_try:
                    # expanded point is even better. Replace x_N
                    vertices[-1] = x_exp
                    func_vals[-1] = f_exp
                else:
                    # x_try was good, x_exp is not better
                    vertices[-1] = x_try
                    func_vals[-1] = f_try
            else:
                # Better than x_N, not better than x_0. Just accept the point
                vertices[-1] = x_try
                func_vals[-1] = f_try
    
        else:
            
            # This point is worse than what we had. First try contracting, new x_try
            x_try = 0.5*(centroid+vertices[-1])
            f_try = func(x_try)
            if f_try < func_vals[-1]:
                # contracting improved x
                vertices[-1] = x_try
                func_vals[-1] = f_try
            else:
                # Nothing worked, just contract all points towards the best points
                vertices = 0.5*(vertices[0] + vertices) # x0 doesnt shift here: 0.5(x0 + x0) = x0
                # need to evaluate all but x0 because they shifted
                func_vals[1:] = func(vertices[1:].T)  

    print('Maximum Number of Evaluations Reached')
    return vertices[0], i            


# CODE FOR THE QUASI-NEWTON METHOD #
def line_minimization(func, x_vec, step_direction, method=golden_section_search):
    """"""
    # Make a function f(x+lmda*n)
    minim_func = lambda lmda: func(x_vec + lmda * step_direction)

    inv_stepdirection = 1./step_direction[0]
    bracket_edge_guess = [1e-3 * inv_stepdirection, 1e3*inv_stepdirection] # This makes a big bracket around a step of ~1
    bracket, _ = make_bracket(minim_func, bracket_edge_guess)
     
    # Use a 1-D minimization method to find the 'best' lmda
    minimum, _ = method(minim_func, bracket)
    min_scipy = fmin(minim_func, [bracket[1]]) # my method doesn't work properly yet
    print(minimum, min_scipy)
    
    #return minimum
    return min_scipy


def compute_gradient(func, x_vec):
    """Computes the gradient of a multi-dimensional function by applying
    Ridder's method on each dimension separately"""
    dim = x_vec.shape[0]
    nabla_f = np.zeros(dim)
    for i in range(dim):
        # The function below transforms the multi-dimensional function func
        # into a function that only varies along dimension i
        func_1d = lambda xi: func([*x_vec[:i], xi, *x_vec[i+1:]])

        nabla_f[i] = ridders_method(func_1d, [x_vec[i]])[0][0] # we don't store the uncertainty now

    return nabla_f
    

def bfgs_update(H, delta, D):
    """Updates the approximated Hessian using the Broyden–Fletcher–Goldfarb–Shannon
    algorithm, used for optimization with the quasi-Newton method.
    INPUTS:
        H: NxN ndarray, approximation of the Hessian
        delta: N ndarray, last taken optimization step in x_vec
        D: N ndarray, difference between new and old gradients
    OUTPUTS:
        H': NXN ndarray, updated approximation of the Hessian        
    """
    # Pre-compute some values for efficiency and clarity 
    deltaD = delta @ D
    HD = H @ D
    DHD = D @ HD

    u = (delta/deltaD) - (HD/DHD)

    H_update1 = outer_product(delta, delta)/deltaD
    H_update2 = outer_product(HD, HD)/DHD
    H_update3 = DHD * outer_product(u, u)
    return H + H_update1 - H_update2 + H_update3


def quasi_newton(func, start, target_step_acc=1e-3, target_grad_acc=1e-3, max_iter=int(1e3)):
    """"""
    # SETUP
    dim = start.shape[0]
    H = np.eye(dim)
    x_vec = start
    # Do this before the loop because we compute the gradient at x_i+1 in loop i
    gradient = compute_gradient(func, x_vec)
        
    for i in range(max_iter):        
        step_direction = -H @ gradient
        step_size = line_minimization(func, x_vec, step_direction)
        # Make the step
        delta = step_size * step_direction

        x_vec += delta
        # Check if we have converged enough
        if step_size < target_step_acc:
            return x_vec, i
        
        # Compute the gradient at the new point, and check relative convergence
        new_gradient = compute_gradient(func, x_vec)
        if np.abs(np.sum((new_gradient - gradient)/(0.5*(new_gradient+gradient)))) < target_grad_acc:
            return x_vec, i
        
        # If no accuracies are reached yet, sadly we have to continue
        D = new_gradient - gradient    
        gradient = new_gradient
        H = bfgs_update(H, delta, D)

    return x_vec, i
