import numpy as np
import matplotlib.pyplot as plt

def split_list(a):
    """Splits a list in two halves and returns both.
        -If the length of the array is even, both halves are equally long.
        -If the length of the array is odd, the second half is one longer"""
    half = len(a) // 2
    return a[:half], a[half:]


def bisection(x, y, t, M):
    """Applies the bisection algorithm at order M in 1D. Inputs:
        x: The x-values of the data points. Assumed strictly increasing
        y: The y-values of the data points
        t: The x-value of the point to identify sample points for
        M: Number of sample points to return"""
    # Use another variable than x to make sure we don't accidentally mess with
    # the real xdata list due to Python shenanigans
    adjacent_idxs = np.arange(0, len(x))

    # Check if we're dealing with extrapolation
    if t < x[0]:
        print(f"Warning: Extrapolation. {t} lies below x data points")
        return np.arange(0, M+1)
    elif t > x[-1]:
        print(f"Warning: Extrapolation. {t} lies above x data points")
        return np.arange(len(x) - M, len(x) + 1)

    # Keep doing the below steps until we find the two points adjacent to t
    while True:
        left, right = split_list(adjacent_idxs)
        print(left, right)
        if (t > x[left[0]]) & (t <= x[right[0]]):
            # Check if we're exactly on the boundary between two sets
            if t > x[left[-1]]:
                adjacent_idxs = [left[-1], right[0]]
                break
            adjacent_idxs = left
        else:
            adjacent_idxs = right

        if len(adjacent_idxs) < 3:  # then adjacent points are identified
            break

    # Now we want to return the corresponding M points surrounding t
    # This adds one extra point to the right if M is odd
    if len(adjacent_idxs) == 2:
        min_idx = adjacent_idxs[0] - ((M-2)//2)
        max_idx = adjacent_idxs[1] + ((M-2)//2)
    elif adjacent_idxs[0] == 0: # Then we're looking at the left first side
        min_idx = 0
        max_idx = M
    else:  # We should only get here if we're dealing with the last section
        min_idx = adjacent_idxs[0] - M
        max_idx = adjacent_idxs[0]

    print(min_idx, max_idx)
    return np.arange(min_idx, max_idx+1)  # +1 to include max_idx

def linear_interpolator(xdata, ydata, t):
    """Uses linear interpolation to estimate the y-value at points t based on x and y data"""
    y_inter = np.zeros_like(t)
    for i in range(len(t)):
        # 2 because we want linear interpretation
        interp_points = bisection(xdata, ydata, t[i], 2)
        dy = ydata[interp_points[1]] - ydata[interp_points[0]]
        dx = xdata[interp_points[1]] - xdata[interp_points[0]]

        print(f"The Point {t[i]} lies in between {xdata[interp_points[0]]} and {xdata[interp_points[1]]}")

        # y(t) = dy/dx * (t-x_i) + y_i
        y_inter[i] = (dy/dx) * (t[i] - xdata[interp_points[0]]) + ydata[interp_points[0]]

    return y_inter

def poly_interpolator(xdata, ydata, t, M):
    """Interpolates a function using a polynomial interpolator implemented with Neville's Algorithm"""
    y_inter = np.zeros_like(t)
    for i in range(len(t)):
        interp_points = bisection(xdata, ydata, t[i], M)


def main():
    x_data = [1.0000, 4.3333, 7.6667, 11.000, 14.333, 17.667, 21.000]
    y_data = [1.4925, 15.323, 3.2356,-29.472,-22.396, 24.019, 26.863]

    # Interpolation
    t = np.linspace(0, 40, 101)
    y_inter = linear_interpolator(x_data, y_data, t, label='Linear Interpolation')

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()

    ax.scatter(x_data, y_data, c='red', label='Data')
    ax.plot(t, y_inter)

    ax.set_xlabel('Time t')
    ax.set_ylabel('Signal Intensity I(t)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()