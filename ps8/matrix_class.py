#############
#
# Functions for matrix operations
# Mostly based on problem class 2
#
#############

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