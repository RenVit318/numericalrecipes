import numpy as np
import matplotlib.pyplot as plt

def bit_reversal(x):
    """Given an arary of indices with length 2^N, bit reverses
    the indices and returns these reversed indexes. 
        e.g. 10: 1010 -> 0101 : 5
             1 : 0001 -> 1000 : 8 (depends on N)"""
    N = np.log2(x)
    if int(N) != int(np.ceil(N)):
        raise ValueError(f'N {N} is not a power of 2')
    reversed_idxs = np.zeros(N, dtype=int)
    for i in range(N):
        bini = bin(i)[2:] # throw out the '0b' part
        bini = (order - len(bini)) * '0' + bini
        reversed_idxs[i] = int('0b'+ bini[-1::-1], 2)
    return reversed_idxs


def dft_recursive(x_real, x_im):
    """Recursively apply the discrete fourier transform following
    the Cooley-Tukey algorithm. We deal with complex numbers using
    two arrays and trigonometric recurrence for efficiency"""
    N = len(x_real)
    if N > 2:
        # Split the array into even and odds, and apply this algorithm to them    
        x_real[::2], x_im[::2] = dft_recursive(x_real[::2], x_im[::2])
        x_real[1::2], x_im[1::2] = dft_recursive(x_real[1::2], x_im[1::2])
    
    # Define variables for trigonometric recurrence
    theta = 2.*np.pi / N
    alpha = 2.*(np.sin(theta/2.)**2)   
    beta = np.sin(theta)
    cos_k = 1 # the first k is zero  
    sin_k = 0 
    print(theta, alpha, beta)
    for k in range((N//2) - 1):
        t_real, t_im = x_real[k], x_im[k]
        k2 = k + N//2 
        # Compute the product of WNk and Hk        
        prod_real = x_real[k2]*cos_k - x_im[k2]*sin_k
        prod_im = x_real[k2]*sin_k + x_im[k2]*cos_k

        # Combine with the above product
        x_real[k] = t_real + prod_real
        x_real[k2] = t_real - prod_real
        x_im[k] = t_im + prod_im
        x_im[k2] = t_im - prod_im

        # Update cos_k and sin_k
        # This does the calculation one time too often, does it matter?

        cos_k = cos_k - alpha*cos_k - beta*sin_k
        sin_k = sin_k - alpha*sin_k + beta*cos_k

    return x_real, x_im
        

def fft_recursive(x, epsilon=1e-10):
    """Applies the Cooley-Tukey algorithm to apply the fast-fourier
    transform to x using recursive function calls"""
    N = len(x)
    order = np.log2(N)
    
    # Check if N is an order of 2
    if int(order) != int(np.ceil(order)):
        raise ValueError(f'Length of array {N} is not an order of 2. Change the array')
    order = int(order)

    # Start by bit-reversing x to get the 1-el FFT solutions. Could do this faster?
    x_real = x

    # One element FFT is done
    #x_real = x[reversed_idxs]
    
    x_im = np.zeros(N, dtype=np.float64)
    # Recursively call the discrete fourier transform method and get the fourier transform
    x_real, x_im =  dft_recursive(x_real, x_im)

    if (x_im < epsilon).any():
        print('Imaginary array is not empty')
        print(x_im)
    return x_real # throw out the imaginary part


def test_fft():
#    func = lambda x: (2*x + np.sin(2.*np.pi*x/5.) + 3.*np.cos(2.*np.pi*x/2.)) * np.sin(2.*x)
    func = lambda x: np.sin(x)
    x = np.linspace(0, 20, 8)
    xx = np.linspace(0, 20, 250)
    y = func(x)
    yy = func(xx)
    print(type(y))
    y_fft = fft_recursive(func(x))
    y_fft_np = np.fft.fft(func(x))
    plt.stem(y_fft, label='Own FFT')

    plt.legend()
    plt.show()

    plt.stem(y_fft_np, label='Numpy FFT')
    plt.show()

    plt.plot(xx, yy)
    plt.scatter(x, y, c='black')
    plt.show()


def main():
    test_fft()

if __name__ == '__main__':
    main()
