import numpy as np

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

def rng_from_mwc(N, x0=1898567, a=4294957665, return_laststate=False):
    """Sample N values using mwc_base32 with starting value x0
    The given value for a is an optimal seed. We use all 64-bits
    to generate the random number, but only return the last 32
    """
    x = np.zeros(N)
    for i in range(N):
        x0 = mwc_base32(x0, a)
        x[i] = to_int32(x0)
    x /= (2**32 - 1) # this ensures we return U(0,1)
    if return_laststate:
        return x, x0
    else: 
        return x
