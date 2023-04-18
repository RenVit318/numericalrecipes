#################
#
# Implementation of linear algebra functions
# 
#################

import numpy as np

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


def lu_decomposition(coefficients, implicit_pivoting=True, epsilon=1e-10):
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
    if implicit_pivoting:
        row_max_inverse = determine_implicit_pivot_coeff(A)

    # Fix division by zero errors to combat round-off
    A.matrix[np.abs(A.matrix)<epsilon] = epsilon

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
