import numpy as np
import matplotlib.pyplot as plt
from functions.minimization import downhill_simplex


def parabola_min_analytic(a, b, c, fa, fb, fc):
    """Analytically computes the x-value of the minimum of a parabola
    that crosses a, b and c
    """
    top = (b-a)**2 * (fb-fc)  - (b-c)**2 * (fb-fa)
    bot = (b-a) * (fb-fc) - (b-c) * (fb-fa)
    return b - 0.5*(top/bot)

def make_bracket(func, bracket, w=(1.+np.sqrt(5))/2, dist_thresh=100, max_iter=10000):
    """Given two points [a, b], attempts to return a bracket triplet
    [a, b, c] such that f(a) > f(b) and f(c) > f(b).
    Note we only compute f(d) once for each point to save computing time"""
    a, b = bracket
    fa, fb = func(a), func(b)
    direction = 1 # Indicates if we're moving right or left
    if fa < fb:
        # Switch the two points
        a, b = b, a
        fa, fb = fb, fa
        direction = 1 # move to the left

    c = b + direction * (b - a) *w
    fc = func(c)
    
    for i in range(max_iter):
        if fc > fb:
            return np.array([a, b, c])  , i+1
        d = parabola_min_analytic(a, b, c, fa, fb, fc)
        fd = func(d)
        if np.isnan(fd):
            print(f'New point d:{d} gives fd:{fd}. Breaking function')
            return np.array([a,b,c]), i+1
        # We might have a bracket if b < d < c
        if (d>b) and (d<c):
            if fd > fb:
                return np.array([a, b, d]), i+1
            elif fd < fc:
                return np.array([b, d, c]), i+1
            # Else we don't want this d
            #print('no parabola, in between b and c')
            d = c + direction * (c - b) * w
        elif (d-b) > 100*(c-b): # d too far away, don't trust it
            #print('no parabola, too far away')
            d = c + direction * (c - b) * w
        elif d < b:
            pass#print('d smaller than b')

        # we shifted but didn't find a bracket. Go again
        a, b, c = b, c, d
        fa, fb, fc = fb, fc, fd

    print('WARNING: Max. iterations exceeded. No bracket was found. Returning last values')
    return np.array([a, b, c]), i+1

def golden_section_search(func, bracket, target_acc=1e-5, max_iter=int(1e5)):
    """Once we have a start 3-point bracket surrounding a minima, this function iteratively
    tightens the bracket to search of the enclosed minima using golden section search."""
    w = 2. -  (1.+np.sqrt(5))/2 # 2 - golden ratio
    a, b, c = bracket
    fa, fb, fc = func(a), func(b), func(c)
    print(fa, fb, fc)
    for i in range(max_iter):
        # Set new point in the largest interval
        # We do this separately because the bracket propagation can just not be generalized sadly
        if np.abs(c-b) > np.abs(b-a): # we tighten towards the right
            d = b + (c-b)*w
            fd = func(d)
            if fd < fb: # min is in between b and c
                a, b, c = b, d, c
                fa, fb, fc = fb, fd, fc
            else: # min is in between a and d
                a, b, c = a, b, d 
                fa, fb, fc = fa, fb, fd
        else: # we tighten towards the left
            d = b + (a-b)*w
            fd = func(d)
            if fd < fb: # min is in between a and b
                a, b, c = a, d, b
                fa, fb, fc = fa, fd, fb
            else: # min is in between d and c
                a, b, c = d, b, c
                fa, fb, fc = fd, fb, fc            
        
        if np.abs(c-a) < target_acc:
            return [b,d][np.argmin([fb, fd])], i+1 # return the x point corresponding to the lowest f(x)

    print("Maximum Number of Iterations Reached")
    return b, i+1

def load_data(dpath='dataset_LATTE.txt'):
    return np.genfromtxt(dpath)

def logistic_func(X, theta):
    """Estiamte the labels of X given model paramters theta.
    This only works for two object classification"""
    z = np.dot(theta, X.T)
    sigma = 1./(1.+np.exp(-z)) 
    return sigma
    #y_hat[sigma >= 0.5] = 1 
    #return y_hat

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
                    return theta, loss[:i+1]    

                step_size = line_minimization(loss_func, theta, grad)
                print(step_size)
                input()
                theta -= step_size*grad

            # Use a downhill simplex to walk down the loss landscape
            case 'simplex':
                print(theta)
                theta, _ = downhill_simplex(loss_func, theta, eval_separate=True)
                return theta, None
    print('Maximum number of iterations reached.')
    return theta, loss_ar


# LINE MINIMIZATION
def line_minimization(func, x_vec, step_direction, method=golden_section_search, minimum_acc=1e-5):
    """"""
    # Make a function f(x+lmda*n)
    minim_func = lambda lmda: func(x_vec + lmda * step_direction)
    bracket_edge_guess = [0, 1]#inv_stepdirection]  # keeps the steps realatively small to combat divergence
    bracket, _ = make_bracket(minim_func, bracket_edge_guess) # make a 3-point bracket surrounding a minimum
    print(bracket)

    # Use a 1-D minimization method to find the 'best' lmda
    minimum, _ = method(minim_func, bracket, target_acc=minimum_acc)
    return minimum



def make_confusion_matrix(labels, pred):
    """"""
    n_features = int(np.max(labels) + 1)
    mat = np.zeros((n_features, n_features))
    for i in range(n_features):
        true = np.where(labels == i)[0]
        for j in range(n_features): 
            if i == j:
                mat[i][i] = len(np.where(pred[true] == i)[0])
            else:
                mat[i][j] = len(np.where(pred[true] == j)[0])

    return mat
        
def compute_F1_score(mat):
    """Compute the F1 score for a 2D confusion matrix. It is defined as
        F1 = 2 x (precision x recall)/(precision+recall)
       with
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)"""
    precision = mat[1][1] / (mat[1][1] + mat[0][1])
    recall = mat[1][1] / (mat[1][1] + mat[1][0])
    return 2. * (precision * recall) / (precision + recall)


def ida(test):
    ones = np.ones(test.shape[0])
    X = test[:,0]
    Y = test[:,1]
    labels = test[:,2]
    m = len(X)
    n = 2
    lr = 0.1
    print(f'{len(X)} Samples in Total\n{len(X[labels==1])} Bulge Stars')

    # Shift features to be ~N(0, 1)
    features = test[:,:2]
    # Add in more features to make this a second order polynomial
    features = np.stack([X, Y, X*X, Y*Y, X*Y], axis=1)
    print(features.shape)
    
    for j in range(features.shape[1]):
        mean = np.mean(features[:,j])
        std = np.std(features[:,j])
        features[:,j] = (features[:,j] - mean)/std
        
    params, _ = logistic_regression(features, labels, lr=lr, minim_type='constant_step')
    print(f'Parameters: {params}')  
    logi = logistic_func(features, params)
    predictions = np.zeros(len(logi))
    predictions[logi>=0.5] = 1

    conf_mat = make_confusion_matrix(labels, predictions)
    f1 = compute_F1_score(conf_mat)
    print(f'Confusion Matrix\n{conf_mat}')
    print(f'F1: {f1}')
    
    xx = np.linspace(min(X), max(X))
    fig, axs = plt.subplots(1,2,figsize=(7,4))
    axs[0].scatter(X, Y, c=labels)
    axs[0].set_title('True')
    axs[1].scatter(X, Y, c=predictions)
    axs[1].set_title('Predicted')
    for ax in axs:
        ax.set_xlabel(r'$X$')
        ax.set_ylabel(r'$Y$')
    #plt.plot(xx, -xx * params[0]/params[1])
    plt.show()
    

def main():
    test = load_data()
    ida(test)

if __name__ == '__main__':
    main()
