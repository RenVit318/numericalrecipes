import numpy as np
import matplotlib.pyplot as plt
from ancillary import merge_sort, ridders_method
from minimize_1d import golden_section_search, make_bracket

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
    func_vals = func(vertices.T) 

    # Start algorithm
    for i in range(1, max_iter+1): 
        # Sort everything by function value
        sort_idxs = merge_sort(key=func_vals)
        vertices = vertices[sort_idxs]
        func_vals = func_vals[sort_idxs]

        # Check if we have reached our accuracy level by comparing the best and worst function evals.
        accuracy =(np.abs(func_vals[-1] - func_vals[0])/np.abs(0.5*(func_vals[-1] + func_vals[0]))) 

        print(accuracy)


        if accuracy < target_acc:
            print(accuracy, target_acc)
            return vertices[0], i # corresponds to func_vals[0], so the best point

        # Compute the centroid of all but the last (worst) point
        centroid = compute_centroid(vertices[:-1])
        
        # Try out a new points
        x_try = 2.* centroid - vertices[-1]
        f_try = func(x_try)
        
        print('current f vals', func_vals)
        print('f_try = ', f_try)

        if f_try < func_vals[-1]:
            print('reflecting')
            # There is improvement in this step
            if f_try < func_vals[0]:
                
                # We are the best point. Try expanding
                x_exp = 2*x_try - centroid
                f_exp = func(x_exp)
                print('f_exp = ', f_exp)
                if f_exp < f_try:
                    print('expanding')
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
            print('f_contract = ', f_try)
            print('contracting')
            if f_try < func_vals[-1]:
                # contracting improved x
                vertices[-1] = x_try
                func_vals[-1] = f_try
            else:
                print('shrinking')
                # Nothing worked, just contract all points towards the best points
                vertices = 0.5*(vertices[0] + vertices) # x0 doesnt shift here: 0.5(x0 + x0) = x0
                # need to evaluate all but x0 because they shifted
                func_vals[1:] = func(vertices[1:].T)  
        if False:
            x = np.linspace(-3, 3, 250)
            x_grid = np.meshgrid(x, x) # Use this to plot onto a 2D space
            y = func(x_grid)

            plt.imshow(np.log10(y), extent=[-3, 3, -3, 3], origin='lower', cmap='jet')
            plt.colorbar()
            plt.scatter(vertices[:,0], vertices[:,1], c='black', zorder=5)
            plt.show()
    print('Maximum Number of Evaluations Reached')
    return vertices[0], i            

from scipy.optimize import fmin
def line_minimization(func, x_vec, step_direction, method=golden_section_search):
    """"""
    print(step_direction)
    minim_func = lambda lmda: func(x_vec + lmda * step_direction)
    inv_stepdirection = 1./step_direction[0]
    bracket_twopoint = [1e-3 * inv_stepdirection, 1e3*inv_stepdirection] # This makes a big bracket around a step of ~1
    three_bracket, i = make_bracket(minim_func, bracket_twopoint)
     
    print(three_bracket)
    minimum, iterations = method(minim_func, three_bracket)
    print('Num Iter Line Mini.', iterations)
    print(fmin(minim_func, [1000]))
    print(minimum)
    return minimum


def compute_gradient(func, x_vec):
    """Computes the gradient of a multi-dimensional function by applying
    Ridder's method on each dimension separately"""
    dim = x_vec.shape[0]
    nabla_f = np.zeros(dim)
    for i in range(dim):
        # The function below transforms the multi-dimensional function func
        # into a function that only varies along dimension i
        func_1d = lambda xi: func([*x_vec[:i], xi, *x_vec[i+1:]])
        print(ridders_method(func_1d, [x_vec[i]])[0][0])
        nabla_f[i] = ridders_method(func_1d, [x_vec[i]])[0][0] # we don't store the uncertainty now

    return nabla_f

def outer_product(v, w):
    """Compute the outer product of two vectors. This is a matrix A with 
           A_ij = v_i * w_j
    NOTE: This function doesn't assume the vectors are of the same size, and 
    this function is not symmetric (outer(v,w) != outer(w,v))
    """
    A = np.zeros((v.shape[0], w.shape[0]))
    for i in range(v.shape[0]):
        A[i] = v[i] * w
    return A
            
    

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
    # Pre-compute some values  
    print(delta, D)
    deltaD = delta @ D
    HD = H @ D
    DHD = D @ HD
    print('H', H)
    print(deltaD, HD, DHD)
    
    u = (delta/deltaD) - (HD/DHD)
    print(u)
    H_update1 = outer_product(delta, delta)/deltaD
    H_update2 = outer_product(HD, HD)/DHD
    H_update3 = DHD * outer_product(u, u)
    H += H_update1 - H_update2 + H_update3
    print('H', H)
    return H
    

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
        print(step_direction)
        # Make the step
        delta = step_size * step_direction

        x_vec += delta
        # Check if we have converged enough
        #if step_size < target_step_acc:
        #    return x_vec, i
        
        # Compute the gradient at the new point, and check relative convergence
        new_gradient = compute_gradient(func, x_vec)
        #if np.abs(np.sum((new_gradient - gradient)/(0.5*(new_gradient+gradient)))) < target_grad_acc:
        #    return x_vec, i
    
        # If no accuracies are reached yet, sadly we have to continue
        D = new_gradient - gradient    
        gradient = new_gradient
        H = bfgs_update(H, delta, D)

    return x_vec, i

def test_minimization():
    func = lambda x: -np.exp(-x[0]*x[0] - x[1]*x[1])
    #func = lambda x: 100*(x[1]-x[0]*x[0])**2 + (1-x[0])**2
    xmin, xmax = -3, 3
    num_steps = 500
    start_point = np.array([5,4], dtype=np.float64)
#    minimum, iterations = downhill_simplex(func, start_point,target_acc=1e-3)
    minimum, iterations = quasi_newton(func, start_point, max_iter=10)
    print('Num Iterations', iterations)
    print(minimum)

    x = np.linspace(xmin, xmax, num_steps)
    x_grid = np.meshgrid(x, x) # Use this to plot onto a 2D space
    y = func(x_grid)
    plt.scatter(*minimum, zorder=5, c='black')

    plt.imshow(y, extent=[xmin, xmax, xmin, xmax], cmap='jet')
    plt.colorbar()

    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.show()
    

def main():
    test_minimization()


if __name__ == '__main__':
    main()
