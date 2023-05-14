import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy import constants as const
from integrator import LeapFrog

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
        plt.savefig('figures/initial_positions.png', bbox_inches='tight')

    return planets

def make_orbits(planets, N, h, plot=False):
    """"""
    if plot:
        fig1, ax1 = plt.subplots(1,1,figsize=(6,6))
        fig2, ax2 = plt.subplots(1,1,figsize=(6,4))
        i = 1

    print("Starting LeapFrog Algorithm")
    for name, obj in planets.items():
        if name == 'sun':
            ax1.scatter(obj.x0[0], obj.x0[1], marker='*', label='Sun')
            continue # do not simulate the sun to itself
        print(name.capitalize())
        # Apply Leapfrog
        obj.simulate_motion(N, h)
        if plot:
            ax1.scatter(obj.x0[0], obj.x0[1], c=f'C{i}', label=name.capitalize())
            ax1.plot(obj.x[:,0], obj.x[:,1], c=f'C{i}')
            ax2.plot(np.arange(obj.x.shape[0]), obj.x[:,2], c=f'C{i}')
            i += 1

    if plot:
        ax1.set_xlabel(r'$X$ [AU]')
        ax1.set_ylabel(r'$Y$ [AU]')
        ax1.set_title('Star and Planet Orbits over 200 Years')
        ax2.set_xlabel(r'Time in Days')
        ax2.set_ylabel(r'$Z$ [AU]')
        ax2.set_title('Planet z-positions over 200 Years')

        fig1.legend()
        fig1.savefig('figures/orbits.png', bbox_inches='tight')
        fig2.savefig('figures/zplane_png', bbox_inches='tight')

    plt.show()
       

def solar_system_sim():
    object_names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    #object_names = ['sun', 'earth']
    h = 0.5
    N = int((365.25 * 200)/h) # 200yrs in total
    print(N)
#    N = 400 # 200 days with 0.5 day steps

    plot = True
    ################3

    planets = get_pos_vel(object_names, plot=plot)
    make_orbits(planets, N, h, plot=plot)    


def main():
    solar_system_sim()

if __name__ == '__main__':
    main()
