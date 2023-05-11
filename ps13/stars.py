import numpy as np
import matplotlib.pyplot as plt

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
                        loss_func=logistic_loss):
    """Perform logistic regression on features X and labels Y
    X should have shape (m, n); Y should have shape (m)"""
    theta = np.ones(X.shape[1])
    loss = np.inf
    
    for i in range(max_iter):
        loss, grad = loss_func(X, Y, theta, return_gradient=True)
        #print(f'Iteration {i}, loss = {loss}')
        if np.abs(np.max(grad)) < eps:
            print('Gradient reached epsilon threshold')
            print(f'Final Loss = {loss}')
            return theta        
        #print('grad ', grad)
        #print('theta ', theta)
        theta -= lr * grad
        
    print('Maximum number of iterations reached.')
    return theta

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
    lr = 1e-3
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
        
    params = logistic_regression(features, labels, lr=lr)
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
