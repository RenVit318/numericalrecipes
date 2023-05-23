import numpy as np
import matplotlib.pyplot as plt


def fft_nd(x, inverse):
    """Apply the Fourier transform to mulitdimensional data. This can easily be done by performing
    the FFT algorithm along each axis separately consecutively"""
    dim = len(x.shape)
    # Start with dim 0 and work up to the highest dimension
    for i in range(dim):
        for j in range(x.shape[i]):
            if inverse:
                x[i, ...] = ifft(x[i, ...])
            else:
                x[i, ...] = fft(x[i, ...])
    return x

def dft_recursive(x, inverse):
    """Function to be called recursively by the FFT algorithm to perform the DFT on
    subsets of the array following the Danielson-Lanczos lemma. For speed we make use
    of trigonometric recurrence, therefore we never have to compute a complex exponent."""
    N = len(x)
    if N > 2:
        even = dft_recursive(x[::2], inverse)
        odd = dft_recursive(x[1::2], inverse)
        x = np.append(even, odd)
        
    # If we want an iFFT, a -1 should appear in the exponent
    if inverse:
        inv_fac = -1.
    else:
        inv_fac = 1.

    # Define the trig. recurrence variables
    theta = 2.*np.pi/N
    alpha = 2.*(np.sin(theta/2)**2)
    beta = np.sin(theta)
    cos_k = 1. # We start with k = 0; cos(0) = 1
    sin_k = 0. #                      sin(0) = 1

    for k in range(0, N//2):
        k2 = k + N//2 # Index of the 'odd' number
        t = x[k]
        
        Wnk = cos_k + inv_fac*1j*sin_k #np.exp(inv_fac*2.j*np.pi*k/N)
        second_factor = Wnk * x[k2]
        
        # one step of the fourier transform
        x[k] = t + second_factor
        x[k2] = t - second_factor

        # Update trig.
        cos_k_new = cos_k - alpha * cos_k - beta*sin_k
        sin_k_new = sin_k - alpha * sin_k + beta*cos_k
        cos_k, sin_k = cos_k_new, sin_k_new
    
    return x

def fft(x, inverse=False):
    """Apply the FFT algorithm to samples x using the recursive Cooley-Tukey algorithm
    If the length of x is not a power of 2, zeros are appended up to the closest higher
    power of 2. This function returns a complex array.
    If inverse is set to True, a '-' sign is introduced in the exponent of W_N^k"""
    # Check the dimensionality of the incoming data
    if len(x.shape) > 1:
        return fft_nd(x, inverse)

    # Check if N is a power of 2
    N = len(x)
    if (np.log2(N)%1) > 0: # Check if it is not an integer
        diff = int(2**(np.ceil(np.log2(N))) - N) # amount of zeros to add to make N a power of 2
        x = np.append(x, np.zeros(diff))
        N = len(x)

    # Cast x into a complex array so we can store
    x = np.array(x, dtype=np.cdouble)
    x_fft = dft_recursive(x, inverse)

    return x_fft


def ifft(x):
    """Apply the inverse FFT algorithm using the recursive Cooley-Tukey algorithm (see fft).
    This function introduced a '-' sign in the exponent and divides the result by N according
    to the lecture notes. This function mostly exists because it looks nicer."""
    x_fft = fft(x, inverse=True)
    return x_fft/len(x_fft)

def fft_2d():   
    pass

def fft_1d():
    #func = lambda x: (2*x + np.sin(2*np.pi*x/5) + 3 * np.cos(2*np.pi*x/2))*np.sin(2*x)
    #func = lambda x: np.exp(-(x-10)**2)
    xmin = 0
    xmax = 20
    N = 2

    sample_x = np.linspace(xmin, xmax, N)
    sample_y = func(sample_x)

    sample_fft = fft(sample_y)
    sample_ifft = ifft(sample_fft)

    # Plotting
    plt.stem(sample_fft)
    plt.show()

    xx = np.linspace(xmin, xmax, 500)
    yy = func(xx)
    plt.plot(xx, yy)

    plt.plot(sample_x, sample_ifft, marker='o', ls='--')
    plt.show()
    

 
def main():
    fft_2d()

if __name__ == '__main__':
    main()
