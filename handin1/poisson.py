#############
#
# Scripts for Q1 on Hand-in Assignment 1
#
#############

import numpy as np
import scipy.stats
from algebra import factorial, log_factorial


def poisson(lmda, k, dtype_int, dtype_float):
    """Returns the Poisson function with mean lamda, evaluated at the integer point k.

        P_lmda(k) = lmda^k * exp(-lmda) / k!

    If k is too large we compute the function evaluated at k in log-space first, and then return the exponent of the
    result to dodge overflow errors, which occur at k ~ 12. The converted function to log space looks like:

        ln(P_lmda(k)) = k*ln(lmda) - lmda - sum_1^k(ln(i))

    All function calls are wrapped in dtypes to limit the amount of memory usage
    """
    if k > 5:
        res = dtype_float(
            dtype_int(k) * np.log(lmda, dtype=dtype_float) - dtype_float(lmda) - log_factorial(k, dtype=dtype_float))
        res = np.exp(res, dtype=dtype_float)
    else:
        res = dtype_float(((lmda ** k) * np.exp(-lmda, dtype=dtype_float)) / factorial(k, dtype=dtype_int))
    return res


def compute_poisson_values(dtype_int=np.int32, dtype_float=np.float32):
    """"Compute the Poisson values for the points provided in Q1 of hand-in assignment 1.
    For testing purposes we compare our values to those from an official library
    """
    print(log_factorial(1, dtype_float))
    values = [[1, 0], [5, 10], [3, 21], [2.6, 40], [101, 200]]
    poisson_prob_ar_self = np.zeros(len(values))
    poisson_prob_ar_scipy = np.zeros(len(values))
    for i, vals in enumerate(values):
        lmda = dtype_float(vals[0])
        k = dtype_int(vals[1])

        poisson_prob_ar_self[i] = poisson(lmda, k, dtype_int, dtype_float)
        poisson_prob_ar_scipy[i] = scipy.stats.poisson.pmf(k, lmda)

    # Print table in Latex ready format
    table = """$\lamda$ & k & Self & Scipy \\\n\hline\n"""

    for i in range(poisson_prob_ar_self.shape[0]):
        table += f'{values[i][0]} & {values[i][1]} & {poisson_prob_ar_self[i]:.6E} & {poisson_prob_ar_scipy[i]:.6E} \\\n'

    print(table)


def main():
    compute_poisson_values()


if __name__ == '__main__':
    main()
