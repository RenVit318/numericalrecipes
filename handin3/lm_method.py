import numpy as np
from ancillary import ridders_method

def determine_implicit_pivot_coeff(mat):
    """Determines the coefficients for implicit pivotting in Crout's Algorithm. It does this by finding
       the absolute maximum value of each row in the matrix, and storing its inverse.

       NOTE: Requires a Matrix object (this script) as input. This ensures correspondence with row_order
    """
    row_max_inverse = np.zeros(mat.num_rows)
    for i in range(mat.num_rows):
        row = mat.matrix[i]
        row_max = row[np.argmax(np.abs(row))]
        row_max_inverse[i] = 1. / row_max

    return row_max_inverse


def lu_decomposition(coefficients, implicit_pivoting=True, epsilon=1e-13):
    """Decomposes a matrix into:
        -L: A matrix with non-zero elements only in the lower-triangle, and ones on the diagonal
        -U: A matrix with non-zero elements only in the upper-triangle, including the diagonal
       These matrices are presented and stored into one.
       The decomposition is done using Crout's Algorithm
    """
    if type(coefficients) == np.ndarray:
        A = Matrix(values=coefficients)
    else:
        A = coefficients

    # Combat round-off erors to dodge division by zero
    A.matrix[np.abs(A.matrix)<epsilon] = epsilon

    if implicit_pivoting:
        row_max_inverse = determine_implicit_pivot_coeff(A)

    imax_ar = np.zeros(A.num_columns)
    # First pivot the matrix
    for i in range(A.num_columns):
        # A.matrix[i:, i] selects all elements on or below the diagonal
        if implicit_pivoting:
            pivot_candidates = A.matrix[i:, i] * row_max_inverse[i:]
        else:
            pivot_candidates = A.matrix[i:, i]

        pivot_idx = i + np.argmax(np.abs(pivot_candidates))
        imax_ar[i] = pivot_idx
        A.swap_rows(i, pivot_idx)

    for i in range(A.num_columns):
        # A.matrix[i:, i] selects all elements on or below the diagonal
        diag_element = A.matrix[i, i]  # Use to scale alpha factors

        for j in range(i + 1, A.num_rows):  # This leaves a zero at the end, not the best fix this!
            A.matrix[j, i] /= diag_element
            for k in range(i + 1, A.num_rows):  # j+1):
                A.matrix[j, k] -= A.matrix[j, i] * A.matrix[i, k]

    return A


def solve_lineqs_lu(LU, b):
    """"Performs the steps to solve a system of linear equations after a matrix A has been LU decomposed. It 
    does this by first applying forward substitution to solve Ly = b, and then applies backward subsituttion
    to solve Ux = y.
    
    Inputs:
        LU: The decomposed L and U matrices, stored in a single Matrix instance
        b: The constraints of the linear equations, ndarray

    Outputs:
        x: Matrix instance containing the solution such that Ax = b
    """
    if type(b) == np.ndarray:
        x = Matrix(values=b)
    else:
        x = b
    # Begin by swapping the x's in the right order
    x.matrix = x.matrix[LU.row_order]

    # Forward Subsitutions. Solves Ly = b
    for i in range(0, x.num_rows):
        x.matrix[i] -= np.sum(LU.matrix[i, :i] * x.matrix[:i])

    # Backward Substitutions. Solves Ux = y
    for i in range(x.num_rows-1, -1, -1):
        x.matrix[i] = (1./LU.matrix[i,i])*(x.matrix[i] - np.sum(LU.matrix[i, i+1:]*x.matrix[i+1:]))

    return x

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


class Matrix():
    """Matrix Class for linear algebra"""

    def __init__(self, values=None, num_rows=None, num_columns=None, dtype=np.float64):
        """Check inputs and create a corresponding matrix or vector"""
        if values is not None:
            self.num_rows = values.shape[0]
            try:
                self.num_columns = values.shape[1]
            except IndexError:
                self.num_columns = 1
                #print(f'Warning! Values has dim=1. Making vector with shape ({self.num_rows}, {self.num_columns})')
            if type(values) == np.ndarray:
                self.matrix = np.array(values, dtype=dtype)
            else:
                print(f'Datatype of values {type(values)} not recognized. Initializing matrix with zeros.')
                self.matrix = np.zeros((num_rows, num_columns), dtype=dtype)
        else:
            self.num_rows = num_rows
            self.num_columns = num_columns
            self.matrix = np.zeros((num_rows, num_columns))

        # Use row order to track rows that have been shuffled
        self.row_order = np.arange(self.num_rows)

    def swap_rows(self, idx1, idx2):
        """Extract rows from a matrix, and switch them. Track the change in row_order"""
        self.matrix[[idx1, idx2]] = self.matrix[[idx2, idx1]]
        self.row_order[[idx1, idx2]] = self.row_order[[idx2, idx1]]

    def scale_row(self, idx, scalar):
        """Multiply all elements of row {idx} by a factor {scalar}"""
        self.matrix[idx] *= scalar

    def add_rows(self, idx1, idx2, scalar):
        """Add row {idx2} multiplied by scalar to row {idx1}"""
        self.matrix[idx1] += scalar * self.matrix[idx2]


def make_param_func(params, i):
    """Given a list of parameters and an index i, return a function with
    p_i as the variable for use in differentation algorithms"""
    # should be a better way to do this right
    first_half_p = params[:i]
    if not i == len(params)-1: # Avoid indexing errrors
        second_half_p = params[i+1:]            
    else:   
        second_half_p = []
    return lambda p: [*first_half_p, p, *second_half_p]


def make_alpha_matrix(xdata, sigma, func, params,
                      h_start=0.1, dec_factor=2, target_acc=1e-10): #derivative params
    """Make a Matrix object containing the sum of N products of derivatives
    where the element i,j is the product of df/dxi and df/dxj. Each value i 
    can be weighted by its uncertainty sigma if desired. If this is not 
    required one can set sigma = 1 to 'ignore' this step"""
    N = len(xdata) # Number of data points
    M = len(params) # Number of parameters
    A = Matrix(num_columns=M, num_rows=M)

    func_derivatives = np.zeros((M, N))

    # Build up all M derivatives
    for i in range(M):
        param_func = make_param_func(params, i)
        # Adjust Ridders method to do this in one go? Big speed upgrade.
        for j in range(N):
            yp = lambda p: func([xdata[j]], *param_func(p))
            dy_dpi, _ = ridders_method(yp, [params[i]], h_start, dec_factor, target_acc)
            func_derivatives[i][j] = dy_dpi

    # Build up A-matrix
    for i in range(M):
        A.matrix[i][i] = alpha_kl(func_derivatives[i], func_derivatives[i], sigma)
        for j in range(i):
            A.matrix[i][j] = alpha_kl(func_derivatives[i], func_derivatives[j], sigma)
            A.matrix[j][i] = A.matrix[i][j]

    return A


def compute_chi_sq(x, y, sigma, func, params):
    """Compute the chi squared value between N points x, y with
    y uncertainty sigma and a function func with parameters params
    Setting sigma = 1 reduces this to just lesat squares"""
    return np.sum(((y - func(x, *params))**2)/(sigma*sigma))

def compute_chi_sq_likepoisson(x, y, func, params):
    """Compute the chi squared value between N points x, y with
    y uncertainty sigma and a function func with parameters params
    under a Poisson distribution assumption, i.e. \sigma = \mu"""
    mean = func(x, *params)
    return np.sum(((y - mean)**2) / (mean))


def make_nabla_chi2(xdata, ydata, sigma, func, params,
                    h_start=0.1, dec_factor=2, target_acc=1e-5,
                    chisq_like_poisson=False):
    """"""
    M = len(params)
    chisq_derivatives = np.zeros(M)
    for i in range(M):
        param_func = make_param_func(params, i)
        if chisq_like_poisson:
            chi2_func_p = lambda p: compute_chi_sq_likepoisson(xdata, ydata, func, param_func(p))
        else:
            chi2_func_p = lambda p: compute_chi_sq(xdata, ydata, sigma, func, param_func(p))
        dchi_dpi, _ = ridders_method(chi2_func_p, [params[i]], h_start, dec_factor, target_acc)
        chisq_derivatives[i] = dchi_dpi

    return beta_k(chisq_derivatives)


def weigh_A_diagonals(A, lmda):
    """"Weigh the diagonal elements of a square matrix A by a factor (1+lmda)""" 
    if not A.matrix.shape[0] == A.matrix.shape[1]:
        raise ValueError(f"This Matrix object is not square: {A.matrix}")
    for i in range(A.matrix.shape[0]):
        A.matrix[i][i] *= (1. + lmda)
    return A


def alpha_kl(dydp1, dydp2, sigma):
    """"""
    return np.sum((1./(sigma**2.)) * dydp1 * dydp2)

def beta_k(dchi_dp):
    """"""
    return -0.5 * dchi_dp


def levenberg_marquardt(xdata, ydata, sigma, func, guess, linear=True, 
                        w=10, lmda=1e-3, chi_acc=1, max_iter=int(1e2),
                        epsilon = 1e-13, # fit procedure params
                        chisq_like_poisson=False,
                        h_start=0.1, dec_factor=2, target_acc=1e-13):  # derivative params
    """"""
       
    if chisq_like_poisson:
        # sqrt becaues it computes the mean
        sigma = np.sqrt(func(xdata, *guess))
        if np.isnan(sigma).any():
            raise ValueError(f'NaN in sigma {sigma}')
        chi2 = compute_chi_sq_likepoisson(xdata, ydata, func, guess)
    else:
        chi2 = compute_chi_sq(xdata, ydata, sigma, func, guess)
    
    N = len(xdata) # Number of data points
    M = len(guess) # Number of parameters
    b = Matrix(num_columns=1, num_rows=M)
    params = guess

    # Can do this beforehand because the derivatives never change
    # if the functions depend linearly on the parameters
    if linear:
        A = make_alpha_matrix(xdata, sigma, func, params, h_start, dec_factor, target_acc)

    for iteration in range(max_iter):
        if linear:
            A_weighted = copy.deepcopy(A) # ensure no pointing goes towards A
            A_weighted = weigh_A_diagonals(A_weighted, lmda) # Make \alpha_prime
        else:
            A = make_alpha_matrix(xdata, sigma, func, params, h_start, dec_factor, target_acc)  
            # Combat round-off errors and divisions by zero
            A.matrix[np.abs(A.matrix)<epsilon] = epsilon
            A_weighted = weigh_A_diagonals(A, lmda)          


        b.matrix = make_nabla_chi2(xdata, ydata, sigma, func, params, h_start, dec_factor, target_acc)

        # Solve the set of linear equations for \delta p with LU decomposition
        LU = lu_decomposition(A_weighted, implicit_pivoting=True)
        delta_p = solve_lineqs_lu(LU, b).matrix

        # Evaluate new chi^2    
        new_params = params + delta_p.flatten()
        
        if chisq_like_poisson:
            new_sigma = np.sqrt(func(xdata, *new_params)) 
            new_chi2 = compute_chi_sq_likepoisson(xdata, ydata, func, new_params)
        else:
            new_chi2 = compute_chi_sq(xdata, ydata, sigma, func, new_params)
        
        delta_chi2 = new_chi2 - chi2

        if delta_chi2 >= 0 or not np.isfinite(new_chi2): # reject the solution
            lmda = w*lmda
            #print(f'delta = {delta_chi2}. Reject. lambda = {lmda:.2E} chi2 = {chi2}')
            continue

        if np.abs(delta_chi2) < chi_acc:
            return params, new_chi2, iteration+1 # converged!

        # accept the step and make it
        params = new_params
        chi2 = new_chi2
        lmda = lmda/w  
        if chisq_like_poisson:
            sigma = new_sigma
        #print(f'delta = {delta_chi2}. Accept. lambda = {lmda:.2E} chi2 = {chi2}')
       
    print("Max Iterations Reached")
    return params, new_chi2, iteration+1


