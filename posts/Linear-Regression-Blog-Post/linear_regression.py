import numpy as np
import pandas as pd


class LinearRegression:
    
        
    def __init__(self):
        self.w = []
        self.score_history = []
        
    def fit_analytic(self, X, y):
        #updates self weight
        
        #padding the feature matrix
        X = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #finding the weight with the formula to find w_hat
        w_hat = np.linalg.inv(X.T@X)@X.T@y

        #updating weight
        self.w = w_hat
        
        
    def gradient(self, P, q):
        #computes the gradient
        return 2*(P@self.w - q)
    
    def fit_gradient(self, X, y, alpha=0.001, max_epochs=1000):
        #update self weight gradient
        
        #pad X
        X_mod = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #random weight to begin with
        p = X_mod.shape[1]
        self.w = np.random.rand(p)        
        
        #compute P and q to save computational value
        P = X_mod.T@X_mod
        q = X_mod.T@y
        
        for j in range(max_epochs):
            
            #compute gradient
            gradient = self.gradient(P, q)
            
            #attach score to score history
            score = self.score(X, y)
            self.score_history.append(score)
            
            #find new weight
            new_w = self.w - (alpha * gradient)
            
            #update weight
            self.w = new_w
        
        
    def score(self, X, y):
        # returns the accuracy of the analytics
        
        X = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #get y_hat
        y_hat = self.predict(X)

        
        n=X.shape[0]
        
        #calculate y bar
        sum_i = np.sum(y)
        y_bar = (1/n) * sum_i
        
        #calculate coefficient of determination
        denominator = 0
        numerator = 0
        for j in range(n-1):
            numerator += (y_hat[j] - y[j])**2
            denominator += (y_bar - y[j])**2
        
        coef_determ = 1 - (numerator / denominator)
        
        return coef_determ
  
    
    def predict(self, X):
        return X@self.w 