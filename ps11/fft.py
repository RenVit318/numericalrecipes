import numpy as np
import matplotlib.pyplot as plt

def bit_reversal(x):
    """Given an arary of indices with length 2^N, bit reverses
    the indices and returns these reversed indexes. 
        e.g. 10: 1010 -> 0101 : 5
             1 : 0001 -> 1000 : 8 (depends on N)"""
    N = len(x)#np.log2(len(x))
    order = np.log2(N)
    if int(order) != int(np.ceil(order)):
        raise ValueError(f'N {N} is not a power of 2')
    reversed_idxs = np.zeros(N, dtype=int)
    for i in range(N):
        bini = bin(i)[2:] # throw out the '0b' part
        bini = (int(order) - len(bini)) * '0' + bini
        reversed_idxs[i] = int('0b'+ bini[-1::-1], 2)
    return reversed_idxs


def dft_recursive(x_real, x_im, inverse=False):
    """Recursively apply the discrete fourier transform following
    the Cooley-Tukey algorithm. We deal with complex numbers using
    two arrays and trigonometric recurrence for efficiency"""
    if inverse:
        pre_fac = -1
    else:
        pre_fac = 1 
    # Inverted FT has a '-' sign in the exponent. We add it into the W_N^k

    N = len(x_real)
    
    if N > 2:
        # Split the array into even and odds, and apply this algorithm to them    
        even_real, even_im = dft_recursive(x_real[::2], x_im[::2])
        odd_real, odd_im = dft_recursive(x_real[1::2], x_im[1::2])
        x_real = np.append(even_real, odd_real)
        x_im = np.append(even_im, odd_im)

    # Define variables for trigonometric recurrence
    theta = 2.*np.pi / N
    alpha = 2.*(np.sin(theta/2.)**2)   
    beta = np.sin(theta)
    cos_k = 1 # the first k is zero  
    sin_k = 0 

    
    for k in range(N//2):
        t_real, t_im = x_real[k], x_im[k]
        k2 = k + N//2 # index of the odd element

        # Compute the product of WNk and Hk 
        prod_real = pre_fac * (x_real[k2]*cos_k - x_im[k2]*sin_k)
        prod_im = pre_fac * (x_real[k2]*sin_k + x_im[k2]*cos_k)

        # Combine with the above product (+ for even, - for odd
        x_real[k] = t_real + prod_real
        x_real[k2] = t_real - prod_real
        x_im[k] = t_im + prod_im
        x_im[k2] = t_im - prod_im

        # Update cos_k and sin_k using trig. recurrence
        # This does the calculation one time too often, does it matter?
        cos_k_new = cos_k - alpha*cos_k - beta*sin_k
        sin_k_new = sin_k - alpha*sin_k + beta*cos_k
        cos_k, sin_k = cos_k_new, sin_k_new

    return x_real, x_im
        


def fft_recursive(x, inverse=False, epsilon=1e-10):
    """Applies the Cooley-Tukey algorithm to apply the fast-fourier
    transform to x using recursive function calls"""
    N = len(x)
    # Check if N is an order of 2
    if (N%2) > 0:
        raise ValueError(f'Length of array {N} is not an order of 2. Change the array')

    x_real = np.array(x, dtype=np.float64)  
    x_im = np.zeros_like(x_real)

    # Recursively call the discrete fourier transform method and get the fourier transform
    x_real, x_im =  dft_recursive(x_real, x_im, inverse)

    # Check if the imaginary array is empty
    if (np.abs(x_im) > epsilon).any():
        print('Imaginary array is not empty')

    if inverse:
        x_real /= N

    return x_real # throw out the imaginary part


def test_fft():
    func = lambda x: (2*x + np.sin(2.*np.pi*x/5.) + 3.*np.cos(2.*np.pi*x/2.)) * np.sin(2.*x)
    #func = lambda x: np.sin(x)
    #func = lambda x: np.exp(-x*x)
    N = 32
    x = np.linspace(-np.pi, np.pi, N)
    xx = np.linspace(-np.pi, np.pi, 250)
    y = func(x)
    yy = func(xx)

    y_fft = fft_recursive(func(x))
    y_recon = fft_recursive(y_fft, inverse=True)

    plt.stem(y_fft, label='Own FFT')
    plt.legend()
    plt.show()

    y_fft_np = np.fft.fft(func(x))
    print(np.allclose(y_fft, y_fft_np))
    y_recon = fft_recursive(y_fft_np, inverse=True)
    
    plt.stem(y_fft_np, label="Numpy FFT")
    #plt.stem(np.append(y_fft_np[N//2:], y_fft_np[:N//2]), label='Numpy FFT')
    plt.legend()
    plt.show()

    plt.plot(xx, yy)
    plt.plot(x, y_recon, marker='o')
    plt.scatter(x, y, c='black')
    plt.show()
    
    plt.plot(x, func(x)/y_recon)
    plt.show()


def main():
    test_fft()

if __name__ == '__main__':
    main()
