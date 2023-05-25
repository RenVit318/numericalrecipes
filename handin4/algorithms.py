import numpy as np
from minimization import line_minimization, downhill_simplex

# LEAPFROG
class LeapFrog():
    """Class to store positions and velocities for any physical object
    whose motion we want to simulate using leapfrog"""
    def __init__(self, x0, v0, acc_func):
        self.x0 = x0
        self.v0 = v0
        self.acc_func = acc_func
        self.h = None # Only works for constant h at this moment
        self.dim = x0.shape[0]

    def _setup_sim(self, N, h, continue_from_last=False):
        """Simulate the motion under accelatrion func for N timesteps"""
        if (self.h is not None) and (self.h != h):
            raise ValueError(f'Provided h {h} does not match existing h {self.h}')
        self.h = h
        if continue_from_last:
            self.index = self.x.shape[0]
            self.x = np.vstack([self.x, np.zeros([N, self.dim])])
            self.v = np.vstack([self.v, np.zeros([2*N, self.dim])])
            self.a = np.vstack([self.a. np.zeros([N, self.dim])])
        else:
            self.x = np.zeros([N, self.dim])        
            self.v = np.zeros([2*N, self.dim]) # Two times so we can store v_i and v_i+0.5
            self.a = np.zeros([N, self.dim])

            self.x[0] = self.x0
            self.v[0] = self.v0
            self.v[1] = self._kickstart() # Get v at v_i+1/2 using RK4 
            self.index = 1    
    
    def _kickstart(self):
        """Given x0 and v0 and a function for acceleration, get the velocity 
        at v_i+1/2. This is done standardized for RK4, but Euler also works"""
        h_rk4 = 0.5*self.h # take a half step
        # Approximated positions after 0.25 and 0.5 of a step using v0
        x_half = self.x[0] + h_rk4 * self.v[0] 
        x_full = self.x[0] + h_rk4 * self.v[0]

        # Find v_1/2 using RK4 and approximated positions
        k1 = h_rk4 * self.acc_func(self.x[0], self.v[0])
        k2 = h_rk4 * self.acc_func(x_half, self.v[0] + 0.5*k1)
        k3 = h_rk4 * self.acc_func(x_half, self.v[0] + 0.5*k2)
        k4 = h_rk4 * self.acc_func(x_full, self.v[0] + k3)

        one_sixth = 1./6.
        one_third = 1./3.
        return self.v[0] + one_sixth*k1 + one_third*k2 + one_third*k3 + one_sixth*k4 

    def simulate_motion(self, N, h, continue_from_last=False):
        """Simulate motion under the given acceleration fucntion"""
        self._setup_sim(N, h, continue_from_last)

        # Leapfrog algorithm
        for i in range(self.index, self.index + N - 1):
            # index in the velocity array. Offset because we leave space for v_i
            v_idx = (2*i) + 1 
            # currently can't use a = a(x, v) because we don't have v_i
            self.a[i] = self.acc_func(self.x[i-1], self.v[i-2]) 
            self.v[v_idx] = self.v[v_idx-2] + self.h * self.a[i]
            self.x[i] = self.x[i-1] + self.h * self.v[v_idx]

# RUNGE KUTTA 4
class RungeKutta4():
    """Class to store positions and velocities for any physical object
    whose motion we want to simulate using leapfrog"""
    def __init__(self, x0, v0, acc_func):
        self.x0 = x0
        self.v0 = v0
        self.acc_func = acc_func
        self.h = None # Only works for constant h at this moment
        self.dim = x0.shape[0]

    def _setup_sim(self, N, h, continue_from_last=False):
        """Simulate the motion under accelatrion func for N timesteps"""
        if (self.h is not None) and (self.h != h):
            raise ValueError(f'Provided h {h} does not match existing h {self.h}')
        self.h = h
        if continue_from_last:
            self.index = self.x.shape[0]
            self.x = np.vstack([self.x, np.zeros([N, self.dim])])
            self.v = np.vstack([self.v, np.zeros([N, self.dim])])
            self.a = np.vstack([self.a. np.zeros([N, self.dim])])
        else:
            self.x = np.zeros([N, self.dim])        
            self.v = np.zeros([N, self.dim]) 
            self.a = np.zeros([N, self.dim])

            self.x[0] = self.x0
            self.v[0] = self.v0
            self.index = 0       

    def simulate_motion(self, N, h, continue_from_last=False):
        """Simulate motion under the given acceleration fucntion"""
        self._setup_sim(N, h, continue_from_last)

        one_third = 1./3.
        one_sixth = 1./6.

        # Leapfrog algorithm
        for i in range(self.index, self.index + N - 1):
            k1v = self.h * self.acc_func(self.x[i], self.v[i])
            k1x = self.h * self.v[i]
        
            k2v = self.h * self.acc_func(self.x[i]+0.5*k1x, self.v[i]+0.5*k1v)
            k2x = self.h * (self.v[i] + 0.5*k1v)
        
            k3v = self.h * self.acc_func(self.x[i]+0.5*k2x, self.v[i]+0.5*k2v)
            k3x = self.h * (self.v[i] + 0.5*k2v)

            k4v = self.h * self.acc_func(self.x[i]+k3x, self.v[i]+k3v)
            k4x = self.h * (self.v[i] + k3v)

            self.v[i+1] = self.v[i] + one_sixth*k1v + one_third*k2v + one_third*k3v + one_sixth*k4v   
            self.x[i+1] = self.x[i] + one_sixth*k1x + one_third*k2x + one_third*k3x + one_sixth*k4x 

# FFT
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

    if inverse:
        x_fft /= N

    return x_fft

def fft_nd(x, inverse=False):
    """Apply the Fourier transform to mulitdimensional data. This can easily be done by performing
    the FFT algorithm along each axis separately consecutively"""
    dim = len(x.shape)
    func = lambda x: fft(x, inverse)
    # Start with dim 0 and work up to the highest dimension
    for i in range(dim):
        x = np.apply_along_axis(func1d=func, axis=i, arr=x)
    return x

# LOGISTIC REGRESSION
def logistic_func(X, theta):
    """Estiamte the labels of X given model paramters theta.
    This only works for two object classification"""
    z = np.dot(theta, X.T)
    sigma = 1./(1.+np.exp(-z)) 
    return sigma

def logistic_loss(X, Y, theta, hypothesis_func=logistic_func,
                  return_gradient=False):
    """Logistic loss functions for features X, labels Y and
    parameters theta"""
    h_theta = hypothesis_func(X, theta)
    # Vectorized version of the logistic loss
    loss = (-1./len(Y)) * np.sum((Y * np.log(h_theta) + (1. - Y) * np.log(1. - h_theta)))
    if return_gradient:
        grad = np.sum((1./len(Y)) * X.T * (h_theta - Y) , axis=1)
        return loss, grad
    return loss 

def logistic_regression(X, Y, lr=0.1, eps=1e-6, max_iter=int(1e4),
                        cost_func=logistic_loss,
                        minim_type='constant_step'):
    """Perform logistic regression on features X and labels Y
    X should have shape (m, n); Y should have shape (m)"""
    theta = np.ones(X.shape[1])
    loss_ar = np.zeros(max_iter)
    # Define a function where we only have to feed in theta, because X, Y are constant
    loss_func = lambda theta, return_gradient=False: cost_func(X, Y, theta, return_gradient=return_gradient)

    for i in range(max_iter):
        match minim_type:
            # Use a constant learning rate to minimize
            case 'constant_step': 
                loss, grad = loss_func(theta, return_gradient=True)
                loss_ar[i] = loss
                if np.abs(np.max(grad)) < eps:
                    print('Gradient reached epsilon threshold')
                    print(f'Final Loss = {loss}')
                    return theta, loss_ar[:i+1]     
                theta -= lr * grad
    
            # Step along -grad, but use line minimization to find the step size
            case 'line_minim':
                loss, grad = loss_func(theta, return_gradient=True)
                loss_ar[i] = loss  
                if np.abs(np.max(grad)) < eps:   
                    print('Gradient reached epsilon threshold')
                    print(f'Final Loss = {loss}')
                    return theta, loss_ar[:i+1]    

                step_size = line_minimization(loss_func, theta, grad)
                theta += step_size*grad

            # Use a downhill simplex to walk down the loss landscape
            case 'simplex':
                theta, _ = downhill_simplex(loss_func, theta, eval_separate=True)
                return theta, loss
    print('Maximum number of iterations reached.')
    return theta, loss_ar



