import numpy as np
import matplotlib.pyplot as plt

## Global Variables ##
dtype = np.float32

def sinc_np(x):
    return np.sin(x, dtype=dtype)/x

def sinc_pw_expansion(x, order):
    """Power Series Expansion of sinc(x) around x=0"""
    res = dtype(0)
    for i in range(order):
        res += dtype(((-1)**i * (x**(2*i)))/(np.math.factorial(2*i+1)))
    print(f'PW Result: {res} dtype:{type(res)}')
    return res


def test_pw_expansion(x, max_order, plot=False):
    """Compute difference between the power series expansion, and 'true' value
    when we increase the order of the expansion"""
    error = np.zeros(max_order, dtype=dtype)
    for order in range(max_order):
        diff = sinc_pw_expansion(x, order) - sinc_np(x)
        error[order] = diff
        print(f'order {order}: {diff}')
      
    if plot: 
        plt.plot(np.arange(max_order), error)
        plt.xlabel('Max Order')
        plt.ylabel('Power Series - Library')
        plt.title(f'Approximation Error at x = {x}')
        plt.show()


def main():
    test_pw_expansion(x=dtype(2), max_order=250, plot=True)



if __name__ == '__main__':
    main()
        
