"""
Coding in UTF8

Design of a Artificial Perceptron using Numpy
	
    Documentation:
    	Parameters:
            eta: float
                Learning rate: b/w 0.0 -> 1.0
            n_iter: int
                Passes over the dataset (train)
            
        Attributes:
            w_: 1 dim array using numpy
                Weights after fit
            errors_: list
                Number of misclassifications in every epoch.
                
    Fitting training data:
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        
        Training vectors, where n_samples is the number of samples
        n_features is the number of features
        
        y : array-like, shape = [n_samples] Target values.
        
        Returns 
        -------
        self : object
    
"""

import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] = update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        # calculate the net input here:
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
