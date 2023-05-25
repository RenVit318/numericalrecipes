import numpy as np
import matplotlib.pyplot as plt
from plotting import set_styles
from cic import get_densities
from algorithms import fft_nd

def plot_at_zslices(data, savename, cb_label, suptitle,   
                    z_slices=[4.5, 9.5, 11.5, 14.5]):

    fig, axs = plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)

    for ax, z in zip(axs.flatten(), z_slices):
        # Integer z appear at the edges, but in our array we have the centers
        # so z=4.5 occurs at index 4
        im = ax.imshow(data[:,:,int(z)], cmap='jet')
        ax.set_title(rf'$Z$ = {z}')
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
    fig.suptitle(suptitle)
    plt.savefig(f'results/{savename}.png', bbox_inches='tight')

def compute_forces(eps=1e-15):
    # Make density grid and compute the density contrasts
    mean_rho = 1024/(16**3)
    rho = get_densities()
    delta = (rho - mean_rho)/mean_rho # Density contrasts

    # Plot slices at various z
    plot_at_zslices(delta, savename='density_contrast_slices', cb_label='Density Contrast', suptitle='Grid Density Contrast')

    # Apply FFT to \delta to get k^2 \Phi~
    phi_fft = fft_nd(delta) 
    
    # Compute all k = sqrt(k_x^2 + k_y^2 + k_z^2)
    # The below only works for data with equal sized vertices in 3D
    k_vals_sq = (np.arange(phi_fft.shape[0], dtype=np.float64))** 2 
    # Using this notation we force each array along a distinct axis resulting in a cube
    k_cube_sq = k_vals_sq[None, None, :] + k_vals_sq[None, :, None] + k_vals_sq[:, None, None]
    k_cube_sq[k_cube_sq < eps] = eps # Fight divide by zero error by setting zero to ~eps_m
   
    potential = fft_nd(phi_fft/k_cube_sq, inverse=True)

    # Plot the log of the FFT potential
    plot_at_zslices(np.log10(np.abs(phi_fft)), savename='fft_potential', cb_label=r'$\log_{10}(|\tilde{\Phi}|)$', suptitle='FFT-Space Potential')
    # Plot the potential
    plot_at_zslices(np.real(potential), savename='potential_slices', cb_label='Potential', suptitle='Grid Potential')
        
def main():
    set_styles()
    compute_forces()

if __name__ == '__main__':
    main()
