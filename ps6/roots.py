import numpy as np
import matplotlib.pyplot as plt
import sys
from ridder import ridders_method

def bisection(func, bracket, target_acc=1e-8, max_iter=int(1e5)):
    """Given a function f(x) and a bracket [a,b], this function returns
    a value c within that interval for which f(c) ~= 0 using bisection.
    Guaranteed to converge, but slowly.

    Inputs:

    Outputs:
    """
    a, b = bracket
    fa, fb = func(a), func(b) 

    # Test if the bracket is good
    if not (fa*fb) < 0:
        raise ValueError("The provided bracket does not contain a root")
    if np.abs(fa) < target_acc:
        return a, 0
    if np.abs(fb) < target_acc:
        return b, 0
    
    for i in range(max_iter):
        c = 0.5*(a+b)
        fc = func(c)

        if np.abs(fc) < target_acc:
            return c, i

        if (fa*fc) < 0:
            # Then the new interval becomes a, c
            # keep the number of function calls as low as possible
            b, fb = c, fc
        elif (fc*fb) < 0:
            # Then the new interval becomes c, b
            a, fa = c, fc
        else:
            print("Warning! Bracket seems to have diverged")
            return c, i

    print("Maximum number of iterations reached")
    return c, i
        

def secant(func, bracket, target_acc=1e-8, max_iter=int(1e5)):
    """Given a function f(x) and a bracket [a,b], this function returns
    a value c within that interval for which f(c) ~= 0 using the secant
    method. Not guaranteed to converge, but faster than bisection.

    Inputs:

    Outputs:
    """
    a, b = bracket
    fa, fb = func(a), func(b) 

    # Test if the bracket is good
    if not (fa*fb) < 0:
        raise ValueError("The provided bracket does not contain a root")
    if np.abs(fa) < target_acc:
        return a, 0
    if np.abs(fb) < target_acc:
        return b, 0

    for i in range(max_iter):
        c = b + fb * ((b-a)/(fa-fb))
        fc = func(c)

        if np.abs(fc) < target_acc:
            return c, i
        # shift all values 'one step back'
        a, b = b, c

    print("Maximum number of iterations reached")
    return c, i

def false_position(func, bracket, target_acc=1e-8, max_iter=int(1e5)):
    """Given a function f(x) and a bracket [a,b], this function returns
    a value c within that interval for which f(c) ~= 0 using the false
    position method. Guaranteed to converge, slow but faster than bisection.

    Inputs:

    Outputs:
    """
    a, b = bracket
    fa, fb = func(a), func(b) 

    # Test if the bracket is good
    if not (fa*fb) < 0:
        raise ValueError("The provided bracket does not contain a root")
    if np.abs(fa) < target_acc:
        return a, 0
    if np.abs(fb) < target_acc:
        return b, 0
      
    for i in range(max_iter):
        c = b + fb * ((b-a)/(fa-fb))
        fc = func(c)

        if np.abs(fc) < target_acc:
            return c, i

        if (fa*fc) < 0:
            # Then the new interval becomes a, c
            # keep the number of function calls as low as possible
            b, fb = c, fc
        elif (fc*fb) < 0:
            # Then the new interval becomes c, b
            a, fa = c, fc
        else:
            print("Warning! Bracket seems to have diverged")
            return c, i

    print("Maximum number of iterations reached")
    return c, i
    

def newton_raphson(func, x, target_acc=1e-8, max_iter=int(1e5)):
    """Given a function f(x) and a starting point x0, this function returns
    a value c within that interval for which f(c) ~= 0 using the Newton Raphson
    method. This method is prone to diverge, but can converge very quick.

    Inputs:

    Outputs:
    """
    for i in range(max_iter):
        fx = func(x)
        if np.abs(func(x)) < target_acc:
            return x, i

        df_dx = ridders_method(func, [x], 0.1, 2, 1e-10)[0]
        x -= fx/df_dx

    print("Maximum number of iterations reached")
    return x, i


def test_root_algos():
    # Functions and Brackets
    func1 = lambda x: x*(x*(x-6) + 11) - 6
    bracket1 = [2.5, 4.0]
    guess1 = 4.0

    func2 = lambda x: np.tan(np.pi*x) - 6
    bracket2 = [0.0, 0.48]
    guess2 = 0.48
    
    func3 = lambda x: x*(x*x - 2) + 2
    bracket3 = [-2., 0.]
    guess3 = 0.

    func4 = lambda x: np.exp(10*(x-1))
    bracket4 = [0.0, 1.5]
    guess4 = 0.

    functions = [func1, func2, func3, func4]
    brackets = [bracket1, bracket2, bracket3, bracket4]
    guesses = [guess1, guess2, guess3, guess4] 
    methods = [bisection, secant, false_position, newton_raphson]
    names = ["Bisection", "Secant", "False Position", "Newton-Raphson"]
    ######

    for i in range(len(functions)):
        print(f'Function {i+1}')
        x = np.linspace(brackets[i][0]-1, brackets[i][1]+1)
        y = functions[i](x)
        plt.plot(x, y)
        plt.axhline(y=0, c='black', ls='--', alpha=0.6)
        for j in range(len(methods)):
            if j == 3:
                root, iterations = methods[j](functions[i], guesses[i])
            else:
                root, iterations = methods[j](functions[i], brackets[i])

            plt.scatter(root, functions[i](root), label=names[j])
            print(f'{names[j]} |\t x_root, y_root: {root:.2f}, {functions[i](root):.2E}; iterations:{iterations}')
        plt.legend()
        plt.show()
                


def main():
    test_root_algos()

if __name__ == '__main__':
    main()
