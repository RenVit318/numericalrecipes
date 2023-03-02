#############
#
# Standard Algebraic Functions
#
#############

import numpy as np

def factorial(k, dtype=np.int64):
    """Computes the factorial k! for any integer k"""
    if k == 0:
        return dtype(1)
    else:
        prod = dtype(1)
        for i in range(k+1):
            prod *= dtype(k)

        return prod


