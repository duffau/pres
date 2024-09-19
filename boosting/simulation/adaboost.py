import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

EPS = 10**-5

class AdaBoost:
    
    def __init__(self, n_estimators=100, learning_rate=1):
        self.M = n_estimators
        self.learning_rate = learning_rate
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

        for m in range(self.M):
            print(f"Fitting iteration {m}...")
            if m == 0:
                w = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                w = update_weights(w, alpha_m, y, y_pred)
                w = w/sum(w)

            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w)
            y_pred = G_m.predict(X)
            
            self.estimators_.append(G_m) 

            error_m = compute_weighted_error(y, y_pred, w)

            alpha_m = compute_alpha(error_m)*self.learning_rate
            self.alphas_.append(alpha_m)

    def staged_decision_function(self, X):
        y_pred_m = np.zeros(X.shape[0])
        for m in range(self.M):
            y_pred_m += self.estimators_[m].predict(X) * self.alphas_[m]
            yield y_pred_m

    def predict(self, X, M_max=None):
        '''
        Predict using fitted model. Arguments:
        X: independent variables - array-like
        M_max: The maximum number of weak learners to use
        '''
        if M_max is None:
            M_max = self.M

        y_pred_m = np.zeros(X.shape[0])
        for m in range(M_max):
            y_pred_m += self.estimators_[m].predict(X) * self.alphas_[m]

        return np.sign(y_pred_m)

    def staged_predict(self, X):
        y_pred_m = np.zeros(X.shape[0])
        for m in range(self.M):
            y_pred_m += self.estimators_[m].predict(X) * self.alphas_[m]
            yield np.sign(y_pred_m)



def compute_weighted_error(y, y_pred, w):
    '''
    Calculate the weighted error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w: individual weights for each observation   
    '''
    return sum(w * np.not_equal(y, y_pred))/sum(w)

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log(1 - error + EPS) - np.log(error + EPS)

def update_weights(w, alpha, y, y_pred):
    ''' 
    Update individual weights w after a boosting iteration. Arguments:
    w: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    return w * np.exp(alpha * np.not_equal(y, y_pred))


def avg_exp_loss(y, y_pred):
    return np.mean(np.exp(-y*y_pred))