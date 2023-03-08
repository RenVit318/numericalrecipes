#############
#
# Functions for matrix operations
# Mostly based on problem class 2
# All of these functions assume the matrix is a Matrix instance (matrix_class.py)
#
#############

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
    A = Matrix(values=coefficients)
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


def check_solution(A, x, b, epsilon=1e-10):
    """Checks a proposed solution to a system of linear equations by computing Ax - b and checking
       if all elements are below some threshold"""
    return (np.abs(mat_vec_mul(A, x) - b) < epsilon).all()


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


# Combine the two functions below? Move into Matrix object
def mat_vec_mul(mat, vec):
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


def mat_mat_mul(A, B):
    """Computes the product between a matrix of shape MxN and another matrix of shape NxL.

    Inputs:
        A: ndarray of shape MxN
        B: ndarray of shape NxL

    Outputs
        res: The result of A x B, ndarray of shape MxL
             If both M and L are one, the result is a single float

    """

    try:
        equality = (A.shape[0] == B.shape[1])
        if not equality:
            raise ValueError(f"Shape of A ({A.shape}) does not match shape of B ({B.shape}) for matrix multiplication.")
    except IndexError:
        pass

    res = np.zeros((A.shape[1], B.shape[0]))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            for k in range(A.shape[0]):
                res[i, j] += A[i, k] * B[k, j]

    return res


def main():
    pass

if __name__ == '__main__':
    main()
