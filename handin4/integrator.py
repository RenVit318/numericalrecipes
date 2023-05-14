import numpy as np

# TODO UPGRADE DIMENSIONALITY
def RungeKutta4(func, x_start, y_start, x_end, h):
    """Solve a 1-dimensional ODE using the 4th order Runge-Kutta method if we have a start point
    boundary condition"""
    one_sixth = 1./6. # do this once, slightly faster    
    one_third = 1./3.

    x_solved = np.arange(x_start, x_end, h) # the points at which we will evaluate the ODE
    y_solved = np.zeros_like(x_solved)
    y_solved[0] = y_start    

    for i in range(len(x_solved)-1):
        x, y = x_solved[i], y_solved[i] # x_n and y_n. For cleanliness purposes
        # Compute the 4 k values 
        k1 = h * func(x, y)
        k2 = h * func(x + 0.5*h, y + 0.5*k1)
        k3 = h * func(x + 0.5*h, y + 0.5*k1)
        k4 = h * func(x + h, y + k3)
    
        y_solved[i+1] = y + one_sixth*k1 + one_third*k2 + one_third*k3 + one_sixth*k4
        
    
    return x_solved, y_solved

class LeapFrog():
    """Class to store positions and velocities for any physical object
    whose motion we want to simulate using leapfrog"""
    def __init__(self, x0, v0, acc_func, kick_func=RungeKutta4):
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
        self.a[0] = self.acc_func(self.x[0], self.v[0])
        return self.v[0] + 0.5 * self.h * self.a[0]   

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

