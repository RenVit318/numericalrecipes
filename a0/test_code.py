def Q1():
    import numpy as np

    a = np.arange(1, 100)
    mean = np.mean(a)
    std = np.std(a)
    print(f"Mean: {mean:.2f}, Std.: {std:2f}")

def Q2():
    import numpy as np

    a = np.arange(1,100)
    odd = a[::2]
    even = a[1::2]

    print(f"Even | Mean: {np.mean(even):.2f}, Std.: {np.std(even):2f}")
    print(f"Odd  | Mean: {np.mean(odd):.2f}, Std.: {np.std(odd):2f}")

def Q3():
    import numpy as np
    
    a = np.arange(1,100)
    mask1 = (a>=10)*(a<=20)
    mask2 = (a>=45)*(a<=57)
    a = a[~(mask1 | mask2)]

    print(f"Mean: {np.mean(a):.2f}, Std.: {np.std(a):2f}")

def Q4():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0,3,100)
    y = 0.8*np.exp(x) - 2.*x
    
    plt.plot(x,y,label='0.8e^x - 2x')
    plt.axhline(y=np.mean(y), c='black', ls='--', label=f'Mean: {np.mean(y):.2f}')
    plt.legend()
    plt.show()

def Q5():
    import numpy as np

    x = np.linspace(0,3,10)
    exp_estimate = np.zeros_like(x)
    order = 5
    for k in range(order):
        exp_estimate += (x**k)/factorial(k)
    print(x)
    print(exp_estimate)


def factorial(k):
    """Only works for int. Could be done faster with np, but assignment asked for code snippet"""
    if k == 0:
        return 1
    res = 1
    for i in range(k):
        res *= (i+1)
        print(res)
    return res

if __name__ == '__main__':
    Q1()
    Q2()
    Q3()
    Q4()
    Q5()    


