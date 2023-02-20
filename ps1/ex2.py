import timeit
import numpy as np

G = 1e-11
c = 2.97e8
c_inv2 = 1./(c**2.)

M = np.random.normal(1e6, 1e5, 10000)

def R_S(no_divide=False):
    
    if no_divide:
        return 2.*G*M*c_inv2
    else:
        return 2.*G*M/(c**2.)

def test_speed():    
    print(timeit.timeit("R_S()", setup="from __main__ import R_S", number=1))
    print(timeit.timeit("R_S(no_divide=True)", setup="from __main__ import R_S", number=1))
    
    
if __name__ == '__main__':
    test_speed()
    
