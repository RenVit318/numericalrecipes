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
        for i in range(1, k+1):
            prod *= dtype(i)
        return prod

def log_factorial(k, dtype=np.int64):
    if k == 0:
        return dtype(1)
    else:
        logsum = dtype(0)
        for i in range(1, k+1):
            logsum += np.log(i, dtype=dtype)

        return logsum


