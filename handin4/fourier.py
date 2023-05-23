import numpy as np
import matplotlib.pyplot as plt
from plotting import set_styles
from cic import get_densities
from algorithms import fft, fft_nd

def plot_at_zslices(data, savename, cb_label,   
                    z_slices=[4.5, 9.5, 11.5, 14.5]):

    fig, axs = plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)

    for ax, z in zip(axs.flatten(), z_slices):
        # Integer z appear at the edges, but in our array we have the centers
        # so z=4.5 occurs at index 4
        im = ax.imshow(data[:,:,int(z)], cmap='jet')
        ax.set_title(f'z = {z}')
        ax.grid(False)
    # Setup a colorbar for all
    # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    cb = fig.colorbar(im, cax=cbar_ax)
    
    cb.set_label(cb_label)
    for ax in axs[:,0]:
        ax.set_ylabel(r'$Y$')
    for ax in axs[-1,:]:    
        ax.set_xlabel(r'$X$')
    plt.savefig(f'results/{savename}.png', bbox_inches='tight')

def compute_forces():
    # Make density grid and compute the density contrasts
    mean_rho = 1024/(16**3)
    rho = get_densities()
    delta = (rho - mean_rho)/mean_rho # Density contrasts

    # Plot slices at various z
    plot_at_zslices(delta, 'density_contrast_slices', 'Density Contrast')

    # Apply FFT to \delta to get k^2 \Phi~
    k = 1
    phi_fft = fft(delta) / (k**2)
    potential = fft_nd(phi_fft, True)

    # Plot the log of the FFT potential
    plot_at_zslices(np.log10(np.abs(phi_fft)), 'fft_potential', r'$\log_{10}(|\tilde{\Phi}|$')
    # Plot the potential
    plot_at_zslices(np.real(potential), 'potential_slices', 'Potential')
        
def main():
    set_styles()
    compute_forces()

if __name__ == '__main__':
    main()
