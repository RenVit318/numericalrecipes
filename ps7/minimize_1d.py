import numpy as np
import matplotlib.pyplot as plt

def parabola_min_analytic(a, b, c, fa, fb, fc):
    """Analytically computes the x-value of the minimum of a parabola
    that crosses a, b and c
    """
    top = (b-a)**2 * (fb-fc)  - (b-c)**2 * (fb-fa)
    bot = (b-a) * (fb-fc) - (b-c) * (fb-fa)
    return b - 0.5*(top/bot)

def make_bracket(func, bracket, w=(1.+np.sqrt(5))/2, dist_thresh=100, max_iter=10000):
    """Given two points [a, b], attempts to return a bracket triplet
    [a, b, c] such that f(a) > f(b) and f(c) > f(b).
    Note we only compute f(d) once for each point to save computing time"""
    a, b = bracket
    fa, fb = func(a), func(b)
    direction = 1 # Indicates if we're moving right or left
    if fa < fb:
        # Switch the two points
        print('switching points')
        a, b = b, a
        fa, fb = fb, fa
        direction = -1 # move to the left

    c = b + direction * (b - a) *w
    fc = func(c)
    
    for i in range(max_iter):
        if fc > fb:
            return np.array([a, b, c])  , i

        d = parabola_min_analytic(a, b, c, fa, fb, fc)
        fd = func(d)
        # We might have a bracket if b < d < c
        if (d>b) and (d<c):
            if fd > fb:
                return np.array([a, b, d]), i
            elif fd < fc:
                return np.array([b, d, c]), i
            # Else we don't want this d
            print('no parabola, in between b and c')
            d = c + direction * (c - b) * w
        elif (d-b) > 100*(c-b): # d too far away, don't trust it
            print('no parabola, too far away')
            d = c + direction * (c - b) * w
        elif d < b:
            print('d smaller than b')

        # we shifted but didn't find a bracket. Go again
        a, b, c = b, c, d
        fa, fb, fc = fb, fc, fd

    print('WARNING: Max. iterations exceeded. No bracket was found. Returning last values')
    return np.array([a, b, c]), i


def golden_section_search(func, bracket, target_acc=1e-5, max_iter=int(1e5)):
    """Once we have a start 3-point bracket surrounding a minima, this function iteratively
    tightens the bracket to search of the enclosed minima using golden section search."""
    w = 2. -  (1.+np.sqrt(5))/2 # 2 - golden ratio
    a, b, c = bracket
    fa, fb, fc = func(a), func(b), func(c)
    
    for i in range(max_iter):
        # Set new point in the largest interval
        # We do this separately because the bracket propagation can just not be generalized sadly
        if np.abs(c-b) > np.abs(b-a): # we tighten towards the right
            d = b + (c-b)*w
            fd = func(d)
            if fd < fb: # min is in between b and c
                a, b, c = b, d, c
                fa, fb, fc = fb, fd, fc
            else: # min is in between a and d
                a, b, c = a, b, d 
                fa, fb, fc = fa, fb, fd
        else: # we tighten towards the left
            d = b + (a-b)*w
            fd = func(d)
            if fd < fb: # min is in between a and b
                a, b, c = a, d, b
                fa, fb, fc = fa, fd, fb
            else: # min is in between d and c
                a, b, c = d, b, c
                fa, fb, fc = fd, fb, fc            
        
        if np.abs(c-a) < target_acc:
            return [b,d][np.argmin([fb, fd])], i+1 # return the x point corresponding to the lowest f(x)

    print("Maximum Number of Iterations Reached")
    return b, i+1
        
    

def test_minimization():
#    func = lambda x: x**4 + 10*x**3 + 10*(x*2)**2
    func = lambda x: np.cos(x) + 0.05*x*x

    x = np.linspace(-30, 30, 500)
    y = func(x)
    plt.plot(x, y)
    plt.axhline(y=0, c='black', ls='--') 
#    plt.ylim(-1, 3000)
   
    start_bracket = np.array([-10, 10 ])
    plt.scatter(start_bracket, func(start_bracket), label='Start Points')

    bracket, iterations = make_bracket(func, start_bracket)
    plt.scatter(bracket, func(bracket), label='Final Bracket')

    minimum, iter2 = golden_section_search(func, bracket)
    plt.scatter(minimum, func(minimum), label=f'GS Minimum (f(x)={func(minimum):.2E})')
    
    print(f'Num Iterations: {iterations+iter2}')
    plt.legend()
    plt.show()   

    


def main():
    test_minimization()

if __name__ == '__main__':
    main()
