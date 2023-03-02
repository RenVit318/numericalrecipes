#############
#
# Scripts for Q1 on Hand-in Assignment 1
#
#############

import numpy as np
import scipy.stats
from algebra import factorial

def poisson(lmda, k, dtype_int, dtype_float):
    """Returns the Poisson function with mean lamda for the integer k"""
    res = ((lmda**k) * np.exp(-lmda, dtype=dtype_float))/factorial(k, dtype=dtype_int)
    print(type(res))
    return 

def compute_poisson_values(dtype_int=np.int32, dtype_float=np.float32):
    """"Compute the Poisson values for the points provided in Q1 of hand-in assignment 1.
    For testing purposes we compare our values to those from an official library
    """

    values = [[1,0], [5,10], [3,21], [2.6,40], [101,200]]
    poisson_prob_ar_self = np.zeros(len(values))
    poisson_prob_ar_scipy = np.zeros(len(values))
    for i, vals in enumerate(values):
        print(vals)
        lmda = dtype_float(vals[0])
        k = dtype_int(vals[1])
        print(factorial(k, dtype=dtype_int), np.math.factorial(k))
        poisson_prob_ar_self[i] = poisson(lmda, k, dtype_int, dtype_floats)
        poisson_prob_ar_scipy[i] = scipy.stats.poisson.pmf(k, lmda)

    table = """$\lamda$\tk\tSelf\tScipy\n"""
    for i in range(poisson_prob_ar_self.shape[0]):
        table += f'{values[i][0]}\t{values[i][1]}\t{poisson_prob_ar_self[i]:.10f}\t{poisson_prob_ar_scipy[i]:.10f}\n'

    print(table)
        






def main():
    compute_poisson_values()

if __name__ == '__main__':
    main()
