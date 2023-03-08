############
#
# Scripts and functions for Exercise 1 of Problem Set 3
# Solving systems of linear equations
#
############

import numpy as np
from scipy.linalg import lu
epsilon = 1e-10 # Numerical accuracy threshold

class Matrix():
    """Matrix Class"""
    def __init__(self, values=None, num_rows=None, num_columns=None, dtype=np.float64):
        """Check inputs and create a corresponding matrix or vector"""
        if values is not None:
            self.num_rows = values.shape[0]
            try:
                self.num_columns = values.shape[1]
            except IndexError:
                self.num_columns = 1
                print(f'Warning! Values has dim=1. Making vector with shape ({self.num_rows}, {self.num_columns})')
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


def determine_implicit_pivot_coeff(mat):
    """Determines the coefficients for implicit pivotting in Crout's Algorithm. It does this by finding
       the absolute maximum value of each row in the matrix, and storing its inverse.
       
       NOTE: Requires a Matrix object (this script) as input. This ensures correspondence with row_order
    """
    row_max_inverse = np.zeros(mat.num_rows)
    for i in range(mat.num_rows):
        row = mat.matrix[i]
        row_max = row[np.argmax(np.abs(row))]
        row_max_inverse[i] = 1./row_max    
    return row_max_inverse


def gauss_jordan(coefficients, constraints, make_inverse=False):
    """Solves a set of linear equations using the Gauss Jordan method

    If make_inverse is set to True, we store an extra matrix that starts as I and undergoes
    the same steps as A and b. The resulting matrix should be A^-1    
"""
    A = Matrix(values=coefficients)
    b = Matrix(values=constraints)
    if make_inverse:
        I = Matrix(np.eye(A.num_rows, A.num_columns))
        matrices = [A, b, I]
    else:
        matrices = [A, b]
    

    # Loop over the columns
    for i in range(A.num_columns):
        # Find the largest non-zero element and place it on the diagonal
        # We only select the pivot point below the diagonal to not mess
        # With the work done in earlier column (could change this if necessary?)
        pivot_idx = i+np.argmax(A.matrix[i:,i])
        pivot_value = A.matrix[pivot_idx, i]
        if np.abs(pivot_value) < epsilon:
            raise ValueError('Singular Matrix')

        for mat in matrices:
            mat.swap_rows(i, pivot_idx)

        # Loop over all elements in the column
        for j in range(A.num_rows):
            element_value = A.matrix[j, i]
           
            if j == i: # Re-scale the element to 1
                for mat in matrices:
                    mat.scale_row(j, np.float64(1./element_value))
                
                pivot_value = A.matrix[j, i]
                
            else: # Reduce the element to zero by reducing the pivot
                for mat in matrices:
                    mat.add_rows(j, i, np.float64(-element_value/pivot_value))


    if check_solution(coefficients, b.matrix, constraints):
        print('Solution is Correct')
        return b.matrix
    else:
        raise ValueError('Calculated Solution is Incorrect')


def lu_decomposition(coefficients, implicit_pivoting=True):
    """Decomposes a matrix into:
        -L: A matrix with non-zero elements only in the lower-triangle, and ones on the diagonal
        -U: A matrix with non-zero elements only in the upper-triangle, including the diagonal
       These matrices are presented and stored into one. 
       The decomposition is done using Crout's Algorithm 
    """
    A = Matrix(values=coefficients)
    if implicit_pivoting:
        row_max_inverse = determine_implicit_pivot_coeff(A)

    # We need a way to make sure that we do not get stuck at a column where all of the elements
    # below the diagonal are zero. In that case it is going to be impossible to get a proper pivot
    # without messing up all of the work done in earlier columns
    # Note all rows in each column that are zero
    #zero_elements = Matrix(num_rows=A.num_rows, num_columns=A.num_columns)
    #zero_elements.matrix[np.abs(A.matrix) > epsilon] = 1
    #TODO: Use this matrix properly

    imax_ar = np.zeros(A.num_columns)
    ## First pivot the matrix
    for i in range(A.num_columns):
        #A.matrix[i:, i] selects all elements on or below the diagonal
        if implicit_pivoting:
            pivot_candidates = A.matrix[i:,i] * row_max_inverse[i:]
        else:
            pivot_candidates = A.matrix[i:,i]
     
        pivot_idx = i+np.argmax(np.abs(pivot_candidates))
        imax_ar[i] = pivot_idx
        A.swap_rows(i, pivot_idx)

    # Start with 'random' pivotting. Fix this later!. Remove?
    while False:
        bad_pivot = False
        for i in range(A.num_columns):
            
            if not bad_pivot:

                nonzero_elements = np.where(np.abs(A.matrix[:, i]) > epsilon)[0]
                nonzero_elements = nonzero_elements[nonzero_elements >= i]
                if len(nonzero_elements) > 0:
                    pivot_idx = np.random.choice(nonzero_elements)
                    A.swap_rows(i, pivot_idx)
                else:
                    bad_pivot = True

            
        
        for i in range(A.num_columns):
            if not bad_pivot:
                if np.abs(A.matrix[i, i]) < epsilon:
                    bad_pivot = True
        if bad_pivot:
            print("Zero element in matrix. Resetting")
            A = Matrix(values=coefficients)
            continue
        print(f'Matrix succesfully pivotted. Matrix:\n{A.matrix}\n{A.row_order}')
        break
    

    for i in range(A.num_columns):
        # A.matrix[i:, i] selects all elements on or below the diagonal
        diag_element = A.matrix[i,i] # Use to scale alpha factors

        for j in range(i+1, A.num_rows): # This leaves a zero at the end, not the best fix this!
            A.matrix[j,i] /= diag_element
            for k in range(i+1, A.num_rows):#j+1):
                A.matrix[j,k] -= A.matrix[j,i]*A.matrix[i,k]

    return A


def solve_lineqs_lu(LU, b):
    """"

    """

    x = Matrix(values=b)
    # Begin by swapping the x's in the right order
    x.matrix = x.matrix[LU.row_order]

    # Forward Subsitutions. Solves Ly = b
    for i in range(0, x.num_rows):
        x.matrix[i] -= np.sum(LU.matrix[i, :i] * x.matrix[:i])

    # Backward Substitutions. Solves Ux = y
    for i in range(x.num_rows-1, -1, -1):
        x.matrix[i] = (1./LU.matrix[i,i])*(x.matrix[i] - np.sum(LU.matrix[i, i+1:]*x.matrix[i+1:]))

    return x
               
def check_lu_decomposition(LU, A, epsilon=1e-10):
    """"Checks if the LU decompostion algorithm properly did its work by multiplying L and U, and comparing
    it against A. Returns True if the sum of (L*U) - A is smaller than some threshold epsilon"""

    L = Matrix(num_rows=LU.num_rows, num_columns=LU.num_columns)
    U = Matrix(num_rows=LU.num_rows, num_columns=LU.num_columns)

    # Can we fill in L and U simultaneously?
    for i in range(LU.num_columns):
        for j in range(LU.num_rows):
            if i == j:
                L.matrix[j, i] = 1
                U.matrix[j, i] = LU.matrix[j,i]
            elif j > i:
                L.matrix[j, i] = LU.matrix[j,i]
            elif j < i:
                U.matrix[j, i] = LU.matrix[j,i]


    #L_times_U = mat_mat_mul(L.matrix, U.matrix)
    L_times_U = np.matmul(L.matrix, U.matrix)
    print(np.abs(L_times_U))
    for i in range(L.num_rows):
        if not (np.abs(L_times_U[i] - A[LU.row_order[i]]) < epsilon).all():   
            raise ValueError("LU Decomposition Error")
    return True

def matrix_vector_mul(mat, vec):
    """Computes the product between a matrix of shape MxN and a vector of shape Nx1.

    Inputs:
        mat: ndarray of shape MxN
        vec: ndarray of shape Nx1
    
    Outputs
        res: The result of mat x vec, ndarray of shape Mx1

    """
        
    res = np.zeros_like(vec)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            res[i] += mat[i, j] * vec[j]
            
    return res
    
    
def check_solution(A, x, b, epsilon=1e-10):
    """Checks a proposed solution to a system of linear equations by computing Ax - b and checking 
       if it is below some threshold"""
    return (np.abs((matrix_vector_mul(A, x) - b)) < epsilon).any()


def test_linear_equation_solvers():
    coefficients = np.array([[3, 8, 1,-12,-4], 
                             [1, 0, 0, -1, 0],
                             [4, 4, 3,-40, 0],
                             [0, 2, 1, -3,-2], 
                             [0, 1, 0,-12, 0]])

    constraints = np.array([2, 0, 1, 0, 0]) 

    #gj_solution = gauss_jordan(coefficients, constraints, make_inverse=True)
    LU = lu_decomposition(coefficients, implicit_pivoting=True)
    x = solve_lineqs_lu(LU, constraints)
    print(x.matrix )
    print(f'L times U:\n{check_lu_decomposition_old(LU, Matrix(values=coefficients))}')
    print(matrix_vector_mul(coefficients,x.matrix))
    print(check_solution(coefficients, x.matrix, constraints))
    print(np.abs(np.sum(np.matmul(coefficients, x.matrix) - constraints )) < epsilon)
    #check_lu_decomposition(LU, Matrix(coefficients))
    #P, L, U = lu(coefficients)
    return
    print(f'True LU Decomposition:')
    print(L)
    print(U)
    print(P)
    print(LU.row_order)
    

def test_lu_low_dim():
    coefficients = np.array([[4, 3], [6, 3]])
    constraints = np.array([6,4])
    LU = lu_decomposition(coefficients)
    print(LU.matrix)
    x = solve_lineqs_lu(LU, constraints)
    print(check_solution(coefficients, x.matrix, constraints))
    

def main():
    test_linear_equation_solvers()
    #test_lu_low_dim()


if __name__ == '__main__':
    main()
