#############
#
# Functions for function interpolation
# Mostly based on problem class 1
#
#############

import numpy as np


def split_list(a):
    """Splits a list in two halves and returns both.
        -If the length of the array is even, both halves are equally long.
        -If the length of the array is odd, the second half is one longer"""
    half = len(a) // 2
    return a[:half], a[half:]


def bisection(x, y, t, M):
    """Applies the bisection algorithm at order M in 1D.

    Inputs:
        x: The x-values of the data points. Assumed strictly increasing
        y: The y-values of the data points
        t: The x-value of the point to identify sample points for
        M: Number of sample points to return

    Returns:
        interp_points: The M points nearest to t along the x-axis
        extrapolated: boolean, True if t lies outside the given range of the x data
    """
    # Use another variable than x to make sure we don't accidentally mess with
    # the real xdata list due to Python shenanigans
    adjacent_idxs = np.arange(0, len(x))

    # Check if we're dealing with extrapolation
    if t < x[0]:
        #print(f"Warning: Extrapolation. {t} lies below x data points")
        return np.arange(0, M), True
    elif t > x[-1]:
        #print(f"Warning: Extrapolation. {t} lies above x data points")
        return np.arange(len(x) - M, len(x)), True

    # Keep doing the below steps until we find the two points adjacent to t
    while True:
        left, right = split_list(adjacent_idxs)
        if (t >= x[left[0]]) & (t < x[right[0]]):
            # Check if we're exactly on the boundary between two sets
            if t > x[left[-1]]:
                adjacent_idxs = [left[-1], right[0]]
                break
            adjacent_idxs = left
        else:
            adjacent_idxs = right

        if len(adjacent_idxs) < 3:  # then adjacent points are identified
            break

    # In the following if-statements we check if the point t is "too close to the edge". This is the
    # case if there are fewer than M/2 points to the left or right of t.
    # If the point is in the middle and M is odd, we add an extra point to the right
    if adjacent_idxs[0] < M//2:
        min_idx = 0
        max_idx = M-1
    elif adjacent_idxs[0] > (len(x) - M):
        min_idx = len(x) - M
        max_idx = len(x) - 1
    else:
        min_idx = int(adjacent_idxs[0] - np.floor((M-2)/2))
        max_idx = int(adjacent_idxs[1] + np.ceil((M-2)/2))

    # Check if bisection worked properly
    if not ((x[min_idx] <= t) & (t <= x[max_idx])):
        raise ValueError(f'Mistake in bisection. {t} not between {x[min_idx]} and {x[max_idx]}')

    interp_points = np.arange(min_idx, max_idx+1)
    return interp_points, False  # +1 to include max_idx


def nevilles_equation(t, P1, P2, x1, x2):
    """"Given two P and x values, computes Neville's Equation. This is used for the polynomial
    interpolation function. The equation is:

    H(x) = ((x_j - x) * F_i(x) + (x - x_i) * G_j(x)) / (x_j - x_i)

    Inputs:
        t : x, point to be interpolated
        P1: F_i(x)
        P2: G_j(x)
        x1: x_i
        x2: x_j

    Outputs:
        H_t: H(x)
    """
    top = (x2 - t) * P1 + (t - x1) * P2
    return top / (x2 - x1)


def poly_interpolator(xdata, ydata, t, M):
    """This function first makes use of the bisection algorithm to identify M data points surrounding
    the point to be interpolated, and then applies Neville's Algorithm to estimate its y-value and
    the uncertainty on that estimation.

    Note: While this function is called 'polynomial interpolator', it does accept values outside of the
          provided range of xdata, and therefore can extrapolate. But the reported accuracy of the
          estimation drops off quickly outside the range.

    Inputs:
        xdata: N data points along the x-axis
        ydata: N data points along the y-axis
        t    : The points along the x-axis we want to interpolate, should always be provided as a
               list or ndarray for indexing purposes
        M    : The order of the polynomial to fit

    Outputs:
        y_inter:   Array of the estimated y-values corresponding to t
        unc_inter; Array of the estimated uncertainties on y_inter. These are estimated as the absolute
                   difference between P_array[0] just after, and just before the final loop.
"""
    y_inter = np.zeros_like(t)
    unc_inter = np.zeros_like(t)
    for i in range(len(t)):
        interp_points, interpolated = bisection(xdata, ydata, t[i], M)
        P_array = np.array(ydata)[interp_points]
        x_points = np.array(xdata)[interp_points]

        for k in range(1, M):
            for j in range(M - k):
                P_array[j] = nevilles_equation(t[i], P_array[j], P_array[j + 1], x_points[j], x_points[j + k])
            if k == M - 2:
                previous_val = P_array[0]  # This is e.g. P012, use its diff. with P0123 as dy
        y_inter[i] = P_array[0]
        unc_inter[i] = np.abs(P_array[0] - previous_val)
    return y_inter, unc_inter