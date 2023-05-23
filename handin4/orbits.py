import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy import constants as const
from plotting import set_styles

from algorithms import LeapFrog, RungeKutta4


# TODO update gravity
def grav_force(r, Rsun, Msun=const.M_sun.value,
               G=const.G.to_value((u.AU**3)*(u.kg**(-1))*(u.d**(-2)))):
    """Compute the N-D gravitational force acting on an object with mass m1
    from an object with mass m2 located at the origin"""
    r_diff = r - Rsun
    return - (G * Msun * r_diff) / (np.power(np.linalg.norm(r_diff), 3))

    #return (G * Msun / (r_diff*r_diff)) * (r_diff/np.sum(np.sqrt(r_diff*r_diff)))             
           
def get_pos_vel(object_names,
                time="2021-12-07 10:00",
                plot=False):
    """Get the positions in AU and velocities in AU/d of all solar system objects at time."""
    t = Time(time)
    planets = {} # Dictionary in which we save our planet objects (and the sun)

    if plot:
        fig, [ax1, ax2] = plt.subplots(1,2,figsize=(8,4), tight_layout=True)
    
    for obj in object_names:
        # get the astropy data
        with solar_system_ephemeris.set('jpl'):
            obj_pos_vel = get_body_barycentric_posvel(obj, t)

        x0 = obj_pos_vel[0].xyz.to_value(u.AU)
        v0 = obj_pos_vel[1].xyz.to_value(u.AU/u.d)
        
        if obj == 'sun':
            r_sun = x0 # Position of the sun, which we take to remain cosntant
           
        # Feed r_sun into the grav_force function to get the correct coordinates
        acc_func = lambda x, v: grav_force(x, r_sun)
        planets[obj] = LeapFrog(x0, v0, acc_func)
        #planets[obj] = RungeKutta4(x0, v0, acc_func)
        
        if plot:
            if obj == 'sun':
                m = '*'
            else:
                m = 'o'
            ax1.scatter(x0[0], x0[1], marker=m)
            ax2.scatter(x0[0], x0[2], label=obj.capitalize(), marker=m)

    if plot:
        ax1.set_xlabel(r'$X$ [AU]')
        ax1.set_ylabel(r'$Y$ [AU]')
        ax2.set_xlabel(r'$X$ [AU]')
        ax2.set_ylabel(r'$Z$ [AU]')
        plt.suptitle('Initial Star and Planet Positions')

        plt.legend()
        plt.savefig('results/initial_positions.png', bbox_inches='tight')

    return planets

def make_orbits(planets, N, h, plot=False,
                compare_to_rk4=False):
    """"""
    if plot:
        fig1, axs1 = plt.subplots(1,2,figsize=(15,6))
        fig2, ax2 = plt.subplots(1,1,figsize=(8,6))
        fig3, ax3 = plt.subplots(1,1,figsize=(8,6))
        fig4, axs4 = plt.subplots(1,2,figsize=(15,6))
    i = 1

    for name, obj in planets.items():
        if name == 'sun':
            # Just place it in all corresponding plots
            axs1[0].scatter(obj.x0[0], obj.x0[1], color='yellow', s=50, marker='*', label='Sun')
            axs1[1].scatter(obj.x0[0], obj.x0[1], color='yellow', s=50, marker='*')
            axs4[0].scatter(obj.x0[0], obj.x0[1], color='yellow', s=50, marker='*', label='Sun')
            axs4[1].scatter(obj.x0[0], obj.x0[1], color='yellow', s=50, marker='*')
            continue # do not simulate the sun to itself

        print(name.capitalize())
        # Apply Leapfrog
        obj.simulate_motion(N, h)
        
        if plot:
            axs1[0].scatter(obj.x0[0], obj.x0[1], c=f'C{i}', label=name.capitalize())
            axs1[1].scatter(obj.x0[0], obj.x0[1], c=f'C{i}')
            axs1[0].plot(obj.x[:,0], obj.x[:,1], c=f'C{i}')
            axs1[1].plot(obj.x[:,0], obj.x[:,1], c=f'C{i}')
            ax2.plot(h*np.arange(obj.x.shape[0]), obj.x[:,2], c=f'C{i}')
            
        
        # Code for the bonus question        
        if compare_to_rk4:
            RK4 = RungeKutta4(obj.x0, obj.v0, obj.acc_func)
            RK4.simulate_motion(N, h)
            
            # Plot the difference in x-positions between RK4 and LF
            ax3.plot(h*np.arange(obj.x.shape[0]), np.abs(RK4.x[:,0] - obj.x[:,0]), c=f'C{i}')  
         
            # Plot the orbits
            axs4[0].scatter(RK4.x0[0], RK4.x0[1], c=f'C{i}', label=name.capitalize())
            axs4[1].scatter(RK4.x0[0], RK4.x0[1], c=f'C{i}')
            axs4[0].plot(RK4.x[:,0], RK4.x[:,1], c=f'C{i}')
            axs4[1].plot(RK4.x[:,0], RK4.x[:,1], c=f'C{i}')
                
        i += 1

    if plot:
        # FIGURE 1
        for ax1 in axs1:
            ax1.set_xlabel(r'$X$ [AU]')
            ax1.set_ylabel(r'$Y$ [AU]')
        fig1.suptitle('Star and Planet Orbits over 200 Years with LeapFrog')

        # Right plot zoomed in on the rocky planets
        axs1[1].set_xlim(-2, 2)
        axs1[1].set_ylim(-2, 2)

        # Place legend next to figures
        fig1.subplots_adjust(right=0.85)
        fig1.legend(loc='right')
        fig1.savefig('results/orbits_lf.png', bbox_inches='tight')

        # FIGURE 2
        ax2.set_xlabel(r'Time in Days')
        ax2.set_ylabel(r'$Z$ [AU]')
        ax2.set_title('Planet z-positions over 200 Years with LeapFrog')
        fig2.savefig('results/zplane_png', bbox_inches='tight')

        # FIGURE 3
        ax3.set_xlabel(r'Time in Days')
        ax3.set_ylabel(r'$|X_{\mathrm{RK4}} - X_{\mathrm{LF}}|$ [AU]') 
        #ax3.set_yscale('log')
        ax3.set_title('Positional Differences')
        fig3.savefig('results/rk4_lf_diff.png', bbox_inches='tight')

        # FIGURE 4
        for ax4 in axs4:
            ax4.set_xlabel(r'$X$ [AU]')
            ax4.set_ylabel(r'$Y$ [AU]')
        fig4.suptitle('Star and Planet Orbits over 200 Years with Runge-Kutta 4')

        # Right plot zoomed in on the rocky planets
        axs4[1].set_xlim(-2, 2)
        axs4[1].set_ylim(-2, 2)

        # Place legend next to figures
        fig4.subplots_adjust(right=0.85)
        fig4.legend(loc='right')
        fig4.savefig('results/orbits_rk4.png', bbox_inches='tight')

    plt.show()
       

def solar_system_sim():
    set_styles()
    object_names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

    h = 0.5
    N = int((365.25 * 50)/h) # 200yrs in total
    print(N)

    plot = True
    compare_to_rk4 = True
    ################3

    planets = get_pos_vel(object_names, plot=plot)
    make_orbits(planets, N, h, plot=plot, compare_to_rk4=compare_to_rk4)    


def main():
    solar_system_sim()

if __name__ == '__main__':
    main()
