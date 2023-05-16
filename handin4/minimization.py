import numpy as np

# LINE MINIMIZATION WITH GOLDEN SECTION SEARCH
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
        a, b = b, a
        fa, fb = fb, fa
        direction = 1 # move to the left

    c = b + direction * (b - a) *w
    fc = func(c)
    
    for i in range(max_iter):
        if fc > fb:
            return np.array([a, b, c])  , i+1
        d = parabola_min_analytic(a, b, c, fa, fb, fc)
        fd = func(d)
        if np.isnan(fd):
            print(f'New point d:{d} gives fd:{fd}. Breaking function')
            return np.array([a,b,c]), i+1
        # We might have a bracket if b < d < c
        if (d>b) and (d<c):
            if fd > fb:
                return np.array([a, b, d]), i+1
            elif fd < fc:
                return np.array([b, d, c]), i+1
            # Else we don't want this d
            #print('no parabola, in between b and c')
            d = c + direction * (c - b) * w
        elif (d-b) > 100*(c-b): # d too far away, don't trust it
            #print('no parabola, too far away')
            d = c + direction * (c - b) * w
        elif d < b:
            pass#print('d smaller than b')

        # we shifted but didn't find a bracket. Go again
        a, b, c = b, c, d
        fa, fb, fc = fb, fc, fd

    print('WARNING: Max. iterations exceeded. No bracket was found. Returning last values')
    return np.array([a, b, c]), i+1

def golden_section_search(func, bracket, target_acc=1e-5, max_iter=int(1e5)):
    """Once we have a start 3-point bracket surrounding a minima, this function iteratively
    tightens the bracket to search of the enclosed minima using golden section search."""
    w = 2. -  (1.+np.sqrt(5))/2 # 2 - golden ratio
    a, b, c = bracket
    fa, fb, fc = func(a), func(b), func(c)
    print(fa, fb, fc)
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

def line_minimization(func, x_vec, step_direction, method=golden_section_search, minimum_acc=1e-5):
    """"""
    # Make a function f(x+lmda*n)
    minim_func = lambda lmda: func(x_vec + lmda * step_direction)
    bracket_edge_guess = [0, 1]#inv_stepdirection]  # keeps the steps realatively small to combat divergence
    bracket, _ = make_bracket(minim_func, bracket_edge_guess) # make a 3-point bracket surrounding a minimum
    print(bracket)

    # Use a 1-D minimization method to find the 'best' lmda
    minimum, _ = method(minim_func, bracket, target_acc=minimum_acc)
    return minimum


# CODE FOR THE N-DIMENSIONAL DOWNHILL SIMPLEX
def compute_centroid(A):
    """Compute the centroid of N points in N dimensional space. x should
    be an NxN array of N vectors with N dimensions (in that order). The function
    then returns one ndarray of length N with the centroid coordinates."""
    return (1./A.shape[1]) * np.sum(A, axis=0)


def downhill_simplex(func, start, shift_func=lambda x: x+1, max_iter=int(1e5), target_acc=1e-5,
                     eval_separate=False):
    """Finds the minimum of a function using the downhill simplex method
    INPUT:
        func: A function taking only one variable as input with dimension N
        start: N-Dimensional numpy array where the function starts searching for a minimum
        shift_func: A function taking only one float as input dictating how to mutate the
                    initial simplex vertices

    OUTPUT:
        
    """
    dim = start.shape[0] # = N
    # Store N+1 vertice vectors in this matrix. This ordering fails if we feed it directly to func,
    # but it allows us to choose a vertex as vertices[i]. The function problem we solve by just   
    # transposing this this matrix.
    vertices = np.zeros((dim+1, dim)) 
    func_vals = np.zeros(dim+1) # Store the f(X) values in here

    # Create the simplex, add slight variation to each vector except the first using 'shift_func'
    vertices[0] = start
    for i in range(dim):
        vertices[i+1] = vertices[0]
        vertices[i+1][i] = shift_func(vertices[i+1][i])

    if eval_separate: # slightly slower, but dodges difficutlies with matrix/vector stuff
        for i in range(dim+1):
            func_vals[i] = func(vertices[i])
    else:
        func_vals = func(vertices.T) 
    # Start algorithm
    for i in range(1, max_iter+1): 
        # Sort everything by function value
        sort_idxs = merge_sort(key=func_vals)
        vertices = vertices[sort_idxs]
        func_vals = func_vals[sort_idxs]

        # Check if we have reached our accuracy level by comparing the best and worst function evals.
        accuracy =(np.abs(func_vals[-1] - func_vals[0])/np.abs(0.5*(func_vals[-1] + func_vals[0]))) 

        if accuracy < target_acc:
            print(accuracy, target_acc)
            return vertices[0], i # corresponds to func_vals[0], so the best point

        # Compute the centroid of all but the last (worst) point
        centroid = compute_centroid(vertices[:-1])
        # Try out a new points
        x_try = 2.* centroid - vertices[-1]
        f_try = func(x_try)
        
        if f_try < func_vals[-1]:
            # There is improvement in this step
            if f_try < func_vals[0]:
                # We are the best point. Try expanding
                x_exp = 2*x_try - centroid
                f_exp = func(x_exp)
                if f_exp < f_try:
                    # expanded point is even better. Replace x_N
                    vertices[-1] = x_exp
                    func_vals[-1] = f_exp
                else:
                    # x_try was good, x_exp is not better
                    vertices[-1] = x_try
                    func_vals[-1] = f_try
            else:
                # Better than x_N, not better than x_0. Just accept the point
                vertices[-1] = x_try
                func_vals[-1] = f_try
    
        else:
            
            # This point is worse than what we had. First try contracting, new x_try
            x_try = 0.5*(centroid+vertices[-1])
            f_try = func(x_try)
            if f_try < func_vals[-1]:
                # contracting improved x
                vertices[-1] = x_try
                func_vals[-1] = f_try
            else:
                # Nothing worked, just contract all points towards the best points
                vertices = 0.5*(vertices[0] + vertices) # x0 doesnt shift here: 0.5(x0 + x0) = x0
                # need to evaluate all but x0 because they shifted
                if eval_separate:
                    for i in range(dim):
                        func_vals[i+1] = func(vertices[i+1])    
                    else:
                        func_vals[1:] = func(vertices[1:].T)  

    print('Maximum Number of Evaluations Reached')
    return vertices[0], i            

# SORTING 
def sort_subarrays(a1, a2, k1, k2):
    """Takes two subarrays 1,2 with indices a1, a2 and values k1, k2 and combines these
    into array with indices a such that the k are sorted in ascending order"""
    N1 = len(a1)  
    N2 = len(a2)

    # We built up a new instance of our sorting array. This is not memory efficient
    # TODO: Study the algorithm below to try and find a better method
    a_sorted = np.zeros(N1+N2)
    
    if N1 == 0: 
        return a2
    if N2 == 0:
        return a1
    
    # Walk through the left- and right- sub arrays separately
    idx1 = 0
    idx2 = 0
    while True:
        if k1[idx1] > k2[idx2]: 
            # Then place the second element to the left of the first element and take 
            # one step to the right in the right sub-array
            a_sorted[idx1+idx2] = a2[idx2]
            idx2 += 1

            # Then, if there are elements remaining in the right array, keep
            # placing them to the left as long as they're smaller
            if idx2 < N2: # need this if statement to save us from indexing errors in the while
                while (k1[idx1] > k2[idx2]):
                    a_sorted[idx1+idx2] = a2[idx2]  
                    idx2 += 1

                    if idx2 >= (N2):
                        # No more elements left in the right array, we can fill out with the left array
                        for j in range(idx1, N1):
                            a_sorted[j+idx2] = a1[j]
                        return a_sorted
                       
                # Now the element from the left array is smaller than the first remaining element
                # from the right array, we can safely place it
                a_sorted[idx1+idx2] = a1[idx1]
                idx1 += 1   
            else: 
                # No more elements left in the right sub-array
                for j in range(idx1, N1):
                    a_sorted[j+idx2] = a1[j]
                return a_sorted

        else:  
            a_sorted[idx1+idx2] = a1[idx1]
            idx1 += 1

        # Check if we have reached the end of the left sub-array
        # If we have, fill out the rest of the array with the right sub-array
        if idx1 == N1:
            for j in range(idx2, N2):
                a_sorted[idx1+j] = a2[j]
            return a_sorted



def merge_sort(a=None, key=None):
    """Sorts the array or list using merge sort. This function iteratively
    builds up the array from single elements which are sorted by sort_subarrays
    Note, in principle one should only provide either 'a' or 'key':

    - If 'a' is provided, that array is sorted in ascending order, and returned 
      by this function
    - If 'key' is provided, this function returns the indices corresponding to the
      order in which the array would be sorted
    - If both 'a' and 'key' are provided, this function assumes 'key' has already
      been previously shuffled, and just swaps indices of the preexisting 'a'
   
    RETURNS: 'a': numpy array
    """
    if key is not None:
        key = np.array(key)

    if a is None and key is not None:
        a = np.arange(len(key))  

    a = np.array(a) 
    subsize = 1    
    N = len(a)
    is_sorted = False
    # Build up the array sorting arrays of increasing subsize
    while not is_sorted:
        subsize *= 2
        if subsize > N:
            is_sorted = True # After this iteration, the array is sorted

        for i in range(int(np.ceil(N/subsize))):
            # We need the min(.. , N) to ensure that we do not exceed the length of the 
            # array with our indexing
            subarray1 = a[i*subsize: i*subsize+int(0.5*subsize)] # First half of the interval
            subarray2 = a[i*subsize+int(0.5*subsize): np.min(((i+1)*subsize, N))]
            if key is not None:
                key1, key2 = key[subarray1] , key[subarray2]
                sorted_sub = sort_subarrays(subarray1, subarray2, key1, key2)
            else:
                # we feed in 'subarrayx' twice because if we only sort a, a is its own key
                sorted_sub = sort_subarrays(subarray1, subarray2, subarray1, subarray2)

            a[i*subsize:subsize*(i+1)] = sorted_sub
           
    return a
