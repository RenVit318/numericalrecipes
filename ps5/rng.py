import numpy as np
import matplotlib.pyplot as plt

def pearson(x, y):
    """Mathematical measure to calculate correlation between two arrays of numbers"""
    top = np.mean(x*y) - np.mean(x)*np.mean(y)
    bot = np.sqrt(np.std(x)*np.std(y))
    return top/bot


def lcg(x, a, c, m):
    """(Multiplicative) Linear Congruential Generator, used to generate random numbers
    Not the best method when used on its own. This is specifically a MLCG if c = 0
    -> If x is the previous value in the sequence, then this function returns the next value
    """
    return (a*x + c) % m

def to_int32(x):
    """Takes any integer x, and returns the last 32 bits"""
    binx = bin(np.uint64(x))
    # First two chars are '0b' can ignore these
    if len(binx) > 33:
        bin32 = binx[-32:]
    else: 
        bin32 = (32 - len(binx)) * '0' + binx[2:]
    return int(bin32, 2)

def mwc_base32(x, a):
    """"""
    # Set the first 32 bits to zero
    x = np.uint64(x)
    x = a*(x & np.uint64((2**32 - 1))) + (x >> np.uint64(32))
    return x

def rng_from_mwc(N, x0=1898567, a=4294957665):
    """Sample N values using mwc_base32 with starting value x0
    The given value for a is an optimal seed. We use all 64-bits
    to generate the random number, but only return the last 32
    """
    x = np.zeros(N)
    for i in range(N):
        x0 = mwc_base32(x0, a)
        x[i] = to_int32(x0)
    return x


def test_rng():
    # PARAMS
    N = int(1e4)
    a = 4294957665
    c = 1013904233
    m = 2**32
    x0 = 256481
    #rng = lambda x: lcg(x, a, c, m)
    rng = lambda x: mwc_base32(x, a)
    ##### 
    
    x = rng_from_mwc(N)
    pearson_corr = pearson(x[:-1], x[1:])    

    plt.scatter(x[:-1], x[1:], s=1, label=r'$r_{x_ix_{i+1}}$' + f' = {pearson_corr:.2f}')
    plt.title(f'LCG: a:{a}; c:{c}; m:{m}; x0:{x0}')
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'$x_{i+1}$')
    plt.legend()
    plt.show()

    plt.plot(np.arange(N), x)
    plt.show()
    
    

def main():
    test_rng()


if __name__ == '__main__':
    main()        
