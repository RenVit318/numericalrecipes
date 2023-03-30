import numpy as np
from matrix_class import Matrix

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


def lu_decomposition(coefficients, implicit_pivoting=True):
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

