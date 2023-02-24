import numpy as np
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

    for i in range(A.num_columns):
        # A.matrix[i:, i] selects all elements on or below the diagonal
        if implicit_pivoting:
            pivot_candidates = A.matrix[i:,i] * row_max_inverse[i:]
        else:
            pivot_candidates = A.matrix[i:,i]
        pivot_idx = i+np.argmax(pivot_candidates)

        # This stores i_max in A.row_order
        A.swap_rows(i, pivot_idx)
        diag_element = A.matrix[i,i] # Use to scale alpha factors
        for j in range(i, A.num_rows):
            element_value = A.matrix[j,i]
            element_value /= diag_element
            # Apply subsitution here!
            
   

    print(A.matrix, A.row_order)

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


def check_solution(A, x, b, epsilon=1e-10):
    """Checks a proposed solution to a system of linear equations by computing Ax - b and checking 
       if it is below some threshold"""
    return np.abs(np.sum(A.dot(x) - b)) < epsilon


def test_linear_equation_solvers():
    coefficients = np.array([[3, 8, 1,-12,-4], 
                             [1, 0, 0, -1, 0],
                             [4, 4, 3,-40, 0],
                             [0, 2, 1, -3,-2], 
                             [0, 1, 0,-12, 0]])
    constraints = np.array([2, 0, 1, 0, 0]) 

    #gj_solution = gauss_jordan(coefficients, constraints, make_inverse=True)
    lu_decomposition(coefficients)
    

def main():
    test_linear_equation_solvers()


if __name__ == '__main__':
    main()
