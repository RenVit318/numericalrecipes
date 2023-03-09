import numpy as np
import matplotlib.pyplot as plt


def plot_on_sphere(rng, num_samples):
    #theta = rng(num_samples) * np.pi
    #phi = rng(num_samples) * np.pi * 2.     
    theta = np.arccos(1. - 2*rng(num_samples))
    phi = 2.*np.pi*rng(num_samples)
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)

    theta = rng(num_samples) * np.pi
    phi = rng(num_samples) * np.pi * 2.  

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    ax.scatter(x, y, z)
    plt.show()


def compare_rng():
    plot_on_sphere(np.random.rand, 5000)


def main():
    compare_rng()


if __name__ == '__main__':
    main()        
