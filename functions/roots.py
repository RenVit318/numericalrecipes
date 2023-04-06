#################
#
# Algorithms to find the root of a function
# 
#################

import numpy as np
from .algebra import ridders_method

def false_position(func, bracket, target_x_acc=1e-10, target_y_acc=1e-10, max_iter=int(1e5)):
    """Given a function f(x) and a bracket [a,b], this function returns
    a value c within that interval for which f(c) ~= 0 using the false
    position method. Guaranteed to converge, slow but faster than bisection.
    """
    a, b = bracket
    fa, fb = func(a), func(b)

    # Test if the bracket is good
    if not (fa*fb) < 0:
        raise ValueError("The provided bracket does not contain a root")

    for i in range(max_iter):
        c = b
        c = a - fa*((a-b)/(fa-fb))
        fc = func(c)

        # Check if we made our precisions
        # x-axis
        if np.abs(b-c) < target_x_acc or np.abs(a-c) < target_x_acc:
            return c, i+1
        # relative y-axis
        if np.abs((fc-fb)/fc) < target_y_acc or np.abs((fc-fa)/fc) < target_y_acc:
            return c, i+1

        if (fa*fc) < 0:
            # Then the new interval becomes a, c
            # keep the number of function calls as low as possible
            b, fb = c, fc
        elif (fc*fb) < 0:
            # Then the new interval becomes c, b
            a, fa = c, fc
        else:
            print("Warning! Bracket might have diverged")
            return c, i+1

    print("Maximum number of iterations reached")
    return c, i


def newton_raphson(func, x, target_x_acc=1e-10, target_y_acc=1e-10, max_iter=int(1e5)):
    """Given a function f(x) and a starting point x0, this function returns
    a value c within that interval for which f(c) ~= 0 using the Newton Raphson
    method. This method is prone to diverge, but can converge very quick.
    """
    for i in range(max_iter):
        fx = func(x)
        if np.abs(func(x)) < target_y_acc:
            return x, i

        df_dx = ridders_method(func, [x], 0.1, 2, 1e-10)[0][0]
        delta_x = fx/df_dx

        if np.abs(df_dx) < target_x_acc:
            return x, i
        x -= delta_x
    print("Maximum number of iterations reached")
    return x, i


def FP_NR(func, bracket, target_x_acc=1e-10, target_y_acc=1e-10, fp_accuracy=1e-5, max_iter=int(1e5)):
    """Finds a root of a function by first applying the False Position algorithm
    to get close to the root, and then switches to Newton-Raphson to accurately
    find its value
    """
    x_close, i_fp = false_position(func, bracket, fp_accuracy, fp_accuracy, max_iter)
    root, i_nr = newton_raphson(func, x_close, target_x_acc, target_y_acc, max_iter-i_fp)
    return root, i_fp+i_nr
