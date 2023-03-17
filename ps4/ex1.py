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
        

def integration_test():
    """"""
    h = 0.1 # integration stepsize

    # Test functions to integrate
    x1 = np.linspace(1, 5, int((5-1)/h))
    y1 = x1*x1
        
    x2 = np.linspace(0, np.pi, int((np.pi)/h))
    y2 = np.sin(x2)

    # Simple Trapezoid Rule
    simp_trap1 = simple_trapezoid(x1, y1)
    simp_trap2 = simple_trapezoid(x2, y2)

    # Simpson's Method
    simpson1 = 0
    simpson2 = 0

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
