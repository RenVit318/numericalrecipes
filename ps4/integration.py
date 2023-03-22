import numpy as np
import matplotlib.pyplot as plt

def simple_trapezoid(x, y):
    """Integrate a set of (x_i, y_i) (possibly sampled from a function f(x)
    using the simple trapezoid rule:
        \int f(x)dx = 
    """
    area_sum = 0
    for i in range(len(x)-1):
        area_sum += 0.5*(x[i+1] - x[i]) * (y[i] + y[i+1])
    return area_sum
        

def romberg_integration(func, a, b, order, open_formula=False):
    """Integrate a function using Romberg Integration"""
    # initiate all parameters
    r_array = np.zeros(order)
    h = b - a
    N_p = 1

    # fill in first estimate, don't do this if we cant evaluate at the edges
    if open_formula:
        # First estimate will be with h = (b-a)/2
        start_point = 0
    else:
        r_array[0] = 0.5*h*(func(b) - func(a))
        start_point = 1

    
    # First iterations to fill out estimates of order m
    for i in range(start_point, order):
        delta = h
        h *= 0.5
        x = a + h

        # Evaluate function at Np points
        for j in range(N_p):
            r_array[i] += func(x)
            x += delta
        # Combine new function evaluations with previous 
        r_array[i] = 0.5*(r_array[i-1] + delta*r_array[i])
        N_p *= 2
    
    # Combine all of our estimations to cancel our error terms
    N_p = 1
    for i in range(1,order):
        N_p *= 4
        for j in range(order-i):
            r_array[j] = (N_p*r_array[j+1] - r_array[j])/(N_p-1)

    return r_array[0]
    




def integration_test():
    """"""
    h = 0.1 # integration stepsize

    # Test functions to integrate
    func1 = lambda x: x*x
    x1 = np.linspace(1, 5, int((5-1)/h))
    y1 = func1(x1)
        
    func2 = lambda x: np.sin(x)
    x2 = np.linspace(0, np.pi, int((np.pi)/h))
    y2 = func2(x2)

    # Simple Trapezoid Rule
    simp_trap1 = simple_trapezoid(x1, y1)
    simp_trap2 = simple_trapezoid(x2, y2)

    # Simpson's Method
    simpson1 = romberg_integration(func1, 1, 5, 6 )
    simpson2 = romberg_integration(func2, 0, np.pi, 6)

    print('Simple Trapezoid Rule')
    print(f'Integral 1: {simp_trap1}')
    print(f'Integral 2: {simp_trap2}')
    print()
    print("Simpson's Method")
    print(f'Integral 1: {simpson1}')
    print(f'Integral 2: {simpson2}')

    


def main():
    integration_test()

if __name__ == '__main__':
    main()
