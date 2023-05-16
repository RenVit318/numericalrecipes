import numpy as np
import matplotlib.pyplot as plt
from plotting import set_styles, hist
from algorithms import logistic_regression

def preprocess_data(fname, plot=False, nbins=None):
    data = np.genfromtxt(fname)
    features = data[:,:-1]
    labels = data[:, -1]

    # Rescale the features
    for i in range(features.shape[1]):
        mean = np.mean(features[:,i])
        std = np.std(features[:,i])
        features[:,i] = (features[:,i] - mean)/std

    # Save to a text file
    np.savetxt('results/scaled_features.txt', features)

    if plot:
        fig, axs = plt.subplots(2,2,figsize=(8,8),sharex=False, sharey=False,tight_layout=True)
        for i in range(features.shape[1]):
            ax = axs.flatten()[i]
            binmin = np.min(features[:,i])
            binmax = np.max(features[:,i])
            n, bin_edges, bin_centers = hist(features[:,i], binmin, binmax, nbins, return_centers=True)
            
            ax.step(bin_centers, n, where='mid', lw=3)
            ax.set_title(f'Feature {i}')
            ax.set_xlabel('Rescaled Values')

        for ax in axs[:,0]:
            ax.set_ylabel('Counts')
        plt.savefig('results/scaled_features_dist.png')
        plt.show()

    return features, labels

def classify(features, labels, lr, minim_type):
    # Start with 'simple' classification of two columns using constant step size
    names = [r'$\kappa_{\mathrm{co}}$', 'Color', 'Extended', 'Emission Flux']
    fig1, ax1 = plt.subplots(1,1, figsize=(9,4))

    for i in range(features.shape[1]):
        for j in range(i+1, features.shape[1]):  
            feats = features[:, [i,j]] # First two columns
            params, loss = logistic_regression(feats, labels, lr=lr, minim_type=minim_type)

            # Plot loss curve
            #if np.dtype(loss) != int:
            ax1.step(np.arange(loss.shape[0]), loss, label=f'{names[i]} + \n{names[j]}')
            
            # Plot data
            fig2, ax2 = plt.subplots(1,1,)
            ax2.scatter(feats[:,0], feats[:,1], s=3, c=labels, cmap='seismic')
            
            # Make the 'cutoff line's
            xx = np.linspace(np.min(features[:,0]), np.max(features[:,1]))
            ax2.plot(xx, -params[0]/params[1] * xx, c='black', ls='--', lw=3)
            ax2.set_xlabel(names[i])
            ax2.set_ylabel(names[j])
            ax2.set_title(f'Final Loss = {loss[-1]}')
            fig2.savefig(f'results/2d_fit_{minim_type}_{names[i]}_{names[j]}.png', bbox_inches='tight')

    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Time for 2D Fits')

    # Shrink current axis by 20%
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig1.savefig(f'results/2d_fit_losses_{minim_type}')

def galaxies():    
    set_styles()
    fname = 'galaxy_data.txt'
    plot = False
    nbins = 20

    lr = 0.1
    minim_type = 'line_minim'
    #####

    features, labels = preprocess_data(fname, plot=plot, nbins=nbins)
    classify(features, labels, lr, minim_type)

def main():
    galaxies()

if __name__ == '__main__':
    main()
