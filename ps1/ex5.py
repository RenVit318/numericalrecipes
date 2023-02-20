import numpy as np
import timeit

## Global variables
triangle_x, triangle_y = np.array([1, 3, 2]), np.array([1, 1, 3])
square_x, square_y = np.array([2, 4, 4, 2]), np.array([2, 2, 4, 4])
pent_x, pent_y = np.array([1, 2, 3, 2, 1]), np.array([1, 1, 2, 3, 2])


# Functions
def polygon_area(x, y):
    """Returns the area of a polygon by looping over the simple area summation function"""
    area = 0
    for i in range(len(x)-1):
        area += x[i]*y[i+1] - x[i+1]*y[i]
    return 0.5 * abs(area)

def polygon_area_vectorized(x, y):
    return 0.5 *  np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))

def test_function_a(print_area=False):
    triangle_area = polygon_area(triangle_x, triangle_y)
    square_area = polygon_area(square_x, square_y)
    pent_area = polygon_area(pent_x, pent_y)

    if print_area:
        print(f'triangle area: {triangle_area}')
        print(f'square area: {square_area}')
        print(f'pent. area: {pent_area}')

def test_function_b(print_area=False):
    triangle_area = polygon_area_vectorized(triangle_x, triangle_y)
    square_area = polygon_area_vectorized(square_x, square_y)
    pent_area = polygon_area_vectorized(pent_x, pent_y)

    if print_area:
        print(f'triangle area: {triangle_area}')
        print(f'square area: {square_area}')
        print(f'pent. area: {pent_area}')

def test_implementations():
    print(f'For loops: {timeit.timeit("test_function_a()", setup="from __main__ import test_function_a")}s')
    print(f'For loops: {timeit.timeit("test_function_b()", setup="from __main__ import test_function_b")}s')
if __name__ == '__main__':
    test_implementations()
