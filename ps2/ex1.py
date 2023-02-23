import numpy as np
import matplotlib.pyplot as plt

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
        min_idx = adjacent_idxs[0] - ((M-2)//2)
        max_idx = adjacent_idxs[1] + ((M-2)//2)

    # Check if bisection worked properly
    if not ((x[min_idx] < t) & (t <= x[max_idx])):
        raise ValueError(f'Mistake in bisection. {t} not between {x[min_idx]} and {x[max_idx]}')


    return np.arange(min_idx, max_idx+1), False  # +1 to include max_idx


def linear_interpolator(xdata, ydata, t):
    """Uses linear interpolation to estimate the y-value at points t based on x and y data"""
    y_inter_array = np.zeros_like(t)
    for i in range(len(t)):
        # 2 because we want linear interpretation
        interp_points, extrapolated = bisection(xdata, ydata, t[i], 2)
        dy = ydata[interp_points[1]] - ydata[interp_points[0]]
        dx = xdata[interp_points[1]] - xdata[interp_points[0]]   

        y_inter_array[i] = (dy/dx) * (t[i] - xdata[interp_points[0]]) + ydata[interp_points[0]]

    return y_inter_array

def nevilles_equation(t, P1, P2, x1, x2):
    top = (x2 - t)*P1 + (t-x1)*P2
    return top / (x2 - x1)
    

def poly_interpolator(xdata, ydata, t, M):
    """Interpolates a function using a polynomial interpolator implemented with Neville's Algorithm"""
    y_inter = np.zeros_like(t)
    for i in range(len(t)):
        interp_points, interpolated = bisection(xdata, ydata, t[i], M)
        P_array = np.array(ydata)[interp_points]
        x_points = np.array(xdata)[interp_points]
        print(f'new t {t[i]}')
        print(P_array)
        #approximation = P_array[np.argmin(np.abs(x_points - t[i]))]
        for k in range(M):
            for j in range(M-k-1):
                print(k, j)
                P_array[j] = nevilles_equation(t[i], P_array[j], P_array[j+1], x_points[j], x_points[j+1])
            print(P_array)
            if k == M - 2:
                uncertainty = P_array[0] # This is e.g. P012, use its diff. with P0123 as dy
    
        y_inter[i] = P_array[0]
#        print(y_inter[i], uncertainty - y_inter[i])
    return y_inter


def run_interpolators():
    x_data = [1.0000, 4.3333, 7.6667, 11.000, 14.333, 17.667, 21.000]
    y_data = [1.4925, 15.323, 3.2356,-29.472,-22.396, 24.019, 26.863]
    M_poly = 4

    # Interpolation
    t = np.linspace(0, 21, 101)
    y_linear_inter = linear_interpolator(x_data, y_data, t)
    y_poly_inter = poly_interpolator(x_data, y_data, t, M_poly)

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()

    ax.scatter(x_data, y_data, c='red', label='Data')
    ax.plot(t, y_linear_inter, label='Linear Interpolation')
    ax.plot(t, y_poly_inter, label=f'Polynomial Interoplation M={M_poly}')

    ax.set_xlabel('Time t')
    ax.set_ylabel('Signal Intensity I(t)')
    plt.legend()
    plt.show()


def main():
    run_interpolators()
    #print(nevilles_equation(4.80, 15.323, 3.2356, 4.333, 7.667))

if __name__ == '__main__':
    main()
