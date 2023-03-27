import numpy as np
import matplotlib.pyplot as plt
import time


def selection_sort(a):
    """Sorts the array or list a using selection sort. Input can be a list or a
    1d array. Output is always a 1d array"""
    a = np.array(a)
    for i in range(len(a) - 1):
        val = a[i]
        idx_min = i # current lowest is i
        # Search for the lowest value after i
        for j in range(i+1, len(a)):
            if a[j] < a[idx_min]:
                idx_min = j
        # Swap two values
        if idx_min != i:
            a = swap(a, i, idx_min)

    return a # a is now sorted

def swap(a, idx1, idx2):
    a[idx1], a[idx2] = a[idx2], a[idx1]
    return a

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
        #print(subsize)
        subsize *= 2
        if subsize > N:
           #subsize = N
            is_sorted = True # After this iteration, the array is sorted

        for i in range(int(np.ceil(N/subsize))):
            # We need the max(.. , N) to ensure that we do not exceed the length of the 
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

        
        

def test_sorting_algos():
    to_sort = [38, 27, 43, 3, 9, 82, 10]
    print('Array to sort:', to_sort)
    print('Sorted array :', merge_sort(to_sort))

    xx = np.logspace(0, 6, 6)
    sort_time_merge = np.zeros(6)
    sort_time_selection = np.zeros(6)
    for i,x in enumerate(xx):
        to_sort = np.random.randint(0, 100, int(x))
        print(f'Array to sort (x={x})', to_sort, '\n')
        t0 = time.time()

        #print('Selection sort', selection_sort(to_sort))
        t1 = time.time()
        #print(f'Time Elapsed: {t1-t0:.3E}s\n')
        
        print('Merge sort', merge_sort(to_sort))
        t2 = time.time()
        print(f'Time Elapsed: {t2-t1:.3E}s\n')
        sort_time_selection[i] = t1 - t0
        sort_time_merge[i] = t2 - t0

    plt.scatter(xx, sort_time_selection, label='Selection Sort')
    plt.scatter(xx, sort_time_merge, label='Merge Sort')
    plt.plot(xx, xx*xx, label=r'x$^2$')
    plt.plot(xx, xx*np.log(xx), label=r'x$\log$(x)')

    plt.xlabel('N')
    plt.ylabel('Time to Sort')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

def test_merge():
    to_sort = np.random.randint(0,50, 25)
    print('array to sort', to_sort)
    print('sorted array', selection_sort(to_sort))
    print('sorted array', merge_sort(to_sort))

def test_key_sorting():
    #key = np.array([5, 489, 1e5, 23, 48, 952, 684, 1e3])
    key = np.array([5, 18, 68, 21, 86, 18, 320 ,86])
    print('array to sort', key)
    indxs = merge_sort(key=key)
    print('indices to sort', indxs)
    print('sorted array', key[indxs])

def main():
    #test_sorting_algos()
    #test_merge()
    test_key_sorting()


if __name__ == '__main__':
    main(), 
