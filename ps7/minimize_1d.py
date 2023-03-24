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


#def golden_sectin_search(func, bracket):

    
    
        
    

def test_minimization():
    func = lambda x: x**4 + 10*x**3 + 10*(x*2)**2

    x = np.linspace(-10, 10, 500)
    y = func(x)
    plt.plot(x, y)
    plt.axhline(y=0, c='black', ls='--') 
    plt.ylim(-1, 3000)
   
    start_bracket = np.array([-200, -190])
    plt.scatter(start_bracket, func(start_bracket), label='Start Points')

    bracket, iterations = make_bracket(func, start_bracket)
    plt.scatter(bracket, func(bracket), label='Final Bracket')
    
    print(f'Num Iterations: {iterations}')
    plt.legend()
    plt.show()   

    


def main():
    test_minimization()

if __name__ == '__main__':
    main()
