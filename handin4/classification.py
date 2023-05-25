import numpy as np
import matplotlib.pyplot as plt
from plotting import set_styles, hist
from algorithms import logistic_func, logistic_regression, make_confusion_matrix, compute_f1_score

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

    return features, labels

def classify(features, labels, lr, minim_type,
             p=2):
    # Start with 'simple' classification of two columns using constant step size
    names = [r'$\kappa_{\mathrm{co}}$', 'Color', 'Extended', 'Emission Flux']
    snames = ['kappa', 'color', 'extended', 'emission_flux'] # names for the savefiles
    fig1, ax1 = plt.subplots(1,1, figsize=(9,4))
    table_txt = "" # Fill out for the confusion matrix table

    for i in range(features.shape[1]):
        for j in range(i+1, features.shape[1]):  
            feats = features[:, [i,j]] # First two columns
            # Add in the bias
            feats = np.append(feats, np.ones(feats.shape[0])[:, np.newaxis], axis=1)
            params, loss = logistic_regression(feats, labels, lr=lr, minim_type=minim_type)

            # Make the confusion matrix and compute F1
            logi = logistic_func(feats, params)
            predictions = np.zeros(len(logi))
            predictions[logi>=0.5] = 1 # label = 1 if logi > 0.5 otherwise label = 0
            conf_mat = make_confusion_matrix(labels, predictions)   
            f1 = compute_f1_score(conf_mat)

            # Plot loss curve
            ax1.step(np.arange(loss.shape[0]), loss, label=f'{names[i]} + \n{names[j]}')
            
            # Plot data
            fig2, ax2 = plt.subplots(1,1,)
            ax2.scatter(feats[:,0], feats[:,1], s=3, c=labels, cmap='seismic')
            
            # Make the 'cutoff line'
            xx = np.linspace(np.min(features[:,0]), np.max(features[:,1]))
            ax2.plot(xx, (-params[0]/params[1] * xx) + params[1], c='black', ls='--', lw=3)
            ax2.set_xlabel(names[i])
            ax2.set_ylabel(names[j])
            ax2.set_title(f'Final Loss = {loss[-1]}')
            ax2.set_xlim(np.percentile(feats[:,0], p), np.percentile(feats[:,0], 100-p))
            ax2.set_ylim(np.percentile(feats[:,1], p), np.percentile(feats[:,1], 100-p))
            fig2.savefig(f'results/2d_fit_{minim_type}_{snames[i]}_{snames[j]}.png', bbox_inches='tight')

            # Add text for the table. Format is
            # Feats. TN TP FN FP F1
            conf_mat = np.array(conf_mat, dtype=int)
            table_txt += f'{names[i]} + {names[j]} & {conf_mat[0][0]} & {conf_mat[1][1]} & {conf_mat[1][0]} & {conf_mat[0][1]} & {f1} \\\\\n'

    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Time for 2D Fits')

    # Shrink current axis by 25%
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig1.savefig(f'results/2d_fit_losses_{minim_type}')

    # write to file
    with open(f'results/confmat_tab_{minim_type}.txt', 'w') as f:
        table_txt = table_txt[:-4]
        f.write(table_txt)
        f.close()


def galaxies():    
    set_styles()
    fname = 'galaxy_data.txt'
    plot = True
    nbins = 20

    lr = 0.1
    minim_types = ['constant_step', 'line_minim']
    #####

    features, labels = preprocess_data(fname, plot=plot, nbins=nbins)
    for minim_type in minim_types:
        classify(features, labels, lr, minim_type)

def main():
    galaxies()

if __name__ == '__main__':
    main()
