import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    
    def __init__(self, n_estimators=100):
        self.M = n_estimators
        self.alphas_ = []
        self.estimators_ = []

    def fit(self, X, y):
        '''
        Fit model. Arguments:
        X: independent variables - array-like matrix
        y: target variable - array-like vector
        M: number of boosting rounds. Default is 100 - integer
        '''
        
        self.estimators_ = []
        self.alphas_ = [] 

        # Iterate over M weak classifiers
        for m in range(self.M):
            print(f"Fitting iteration {m}...")
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # (d) Update w_i
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.estimators_.append(G_m) # Save to list of weak classifiers

            # (b) Compute weighted error
            error_m = compute_weighted_error(y, y_pred, w_i)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas_.append(alpha_m)


    def predict(self, X, M_max=None):
        '''
        Predict using fitted model. Arguments:
        X: independent variables - array-like
        M_max: The maximum number of weak learners to use
        '''
        if M_max is None:
            M_max = self.M

        # Predict class label for each weak classifier, weighted by alpha_m
        y_pred_m = np.zeros(X.shape[0])
        for m in range(M_max):
            y_pred_m += self.estimators_[m].predict(X) * self.alphas_[m]

        # Calculate final predictions
        y_pred = (1 * np.sign(y_pred_m)).astype(int)

        return y_pred

    def staged_predict(self, X):
        y_pred_m = np.zeros(X.shape[0])
        for m in range(self.M):
            y_pred_m += self.estimators_[m].predict(X) * self.alphas_[m]
            yield (1 * np.sign(y_pred_m)).astype(int)



def compute_weighted_error(y, y_pred, w_i):
    '''
    Calculate the weighted error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation
    
    Note that all arrays should be the same length
    '''
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration. Arguments:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))