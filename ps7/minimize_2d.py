import numpy as np
import matplotlib.pyplot as plt
from sorting import merge_sort

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



        


def test_minimization():
    #func = lambda x: -np.exp(-x[0]*x[0] - x[1]*x[1])
    func = lambda x: 100*(x[1]-x[0]*x[0])**2 + (1-x[0])**2
    xmin, xmax = -3, 3
    num_steps = 500
    start_point = np.array([0.1,0.1])
    minimum, iterations = downhill_simplex(func, start_point,target_acc=1e-3)
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
