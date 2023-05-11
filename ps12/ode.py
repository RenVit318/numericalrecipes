import numpy as np
import matplotlib.pyplot as plt


def eulers_method_1d_ivp(func, x_start, y_start, x_end, h):
    """Solve a 1-dimensional ODE using Euler's method if we have the initial values"""
    
    x_solved = np.arange(x_start, x_end, h) # the points at which we will evaluate the ODE
    y_solved = np.zeros_like(x_solved)
    y_solved[0] = y_start    

    for i in range(len(x_solved)-1):
        y_solved[i+1] = y_solved[i] + h * func(x_solved[i], y_solved[i])
    
    return x_solved, y_solved


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


def leapfrog(func, x0, v0, h, N, store_vi=False):
    """Use the Leapfrog algorithm to simulate motion"""
    dim = len(x0)
    x = np.zeros((N, dim))
    v = np.zeros((2*N, dim)) # possibly store v_i and v_i+1
    x[0], v[0] = x0, v0

    # First kick
    a = func(x0, v0)
    print(a)
    v[1] = v[0] + 0.5*h * a  # Not correct, use RK4 here  

    for i in range(1,N):
        #if store_vi: Don't really know what to do here
        #    pass
        print(v)
        v[(2*i)+1] = v[(2*(i-1))+1] + h * a
        x[i] = x[i-1] + h * v[(2*i)+1]
        a = func(x[i], v[(2*i)+1])
        print(a)
    return x


def grav_force(m1, r, G = 6.674e-11):
    """Compute the N-D gravitational force acting on an object with mass m1
    from an object with mass m2 located at the origin"""
    return (G * m1 / (r*r)) * (r/np.sum(np.sqrt(r*r)))



def planet_orbits():
    h = 100
    N = 10
    # Constants
    M_J = 9.55e-4 
    G = 6.674e-11 # m^3 kg^-1 s^-2
    yr_to_s = 31556926
    AU_to_m = 149597870700

    ###
    mass_star = 1.989e30 # kg
    # Planet A
    mass_a = M_J * mass_star
    period_a = 12 * yr_to_s # s 
    radius_a = G * (mass_star + mass_a) * period_a**2 / (4*np.pi**2)
    vcirc_a = 2*np.pi*radius_a / period_a
    print(radius_a)
    radius_a /= AU_to_m
    vcirc_a /= (yr_to_s*AU_to_m)
    print(vcirc_a)
    print(radius_a)
    mass_star = 1
    mass_a = M_J
    

    # Planet B
    #mass_b = 0.011 * M_J
    print(radius_a)
    # Place it first at the copmlete right from the star
    x_start = np.array([radius_a, 1])
    print(x_start)
    print(x_start/np.sum(np.sqrt(x_start*x_start)))
    input()
    v_start = np.array([0, vcirc_a])
    orbit_func = lambda x, v: grav_force(mass_star, x)
    x_orbit = leapfrog(orbit_func, x_start, v_start, h, N)

    # Plot
    plt.scatter(0, 0, s=30, marker='*', label='Star')
    plt.scatter(*x_start, s=30, marker='o', label='Planet Start')
    plt.plot(x_orbit[:,0], x_orbit[:,1], marker='o')
    plt.show()    


def test_ode_solvers():
    # Functions and boundary conditions
    func = lambda x, y: -y
    analytic = lambda x: 2*np.exp(-x)
    t = np.linspace(0,20, 1000)
    h = 1e-3

    x_euler, y_euler = eulers_method_1d_ivp(func, t[0],2, t[-1], h)
    x_rk4, y_rk4 = RungeKutta4(func, [t[0]], [2], [t[-1]], h)

    plt.plot(t, analytic(t), label='Analytic Solution', c='black', ls='--')
    plt.plot(x_euler, y_euler, label="Euler's Method")
    plt.plot(x_rk4, y_rk4, label='Runge-Kutta 4')
    plt.legend()
    plt.show()

    # Test accuracy
    plt.scatter(x_euler, np.abs((y_euler - analytic(x_euler))/analytic(x_euler)), label='Euler')
    plt.scatter(x_rk4, np.abs((y_rk4 - analytic(x_rk4))/analytic(x_rk4)), label='RK4')
    plt.yscale('log')
    plt.legend()
    plt.show()





def main():
    #test_ode_solvers()
    planet_orbits()

if __name__ == '__main__':
    main()
