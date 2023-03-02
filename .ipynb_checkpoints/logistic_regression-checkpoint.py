import numpy as np
from scipy.optimize import minimize
np.seterr(all='ignore')


class LogisticRegression:
    
    def __init__(self):
        self.w = []
        self.loss_history = []
        self.score_history = []
        
            
    def predict(self, X):
        #return vector y hat {0,1} of predicted labels
        return X@self.w
        
        
    def score(self, X, y):
        #return accuracy of predictions as number between 0 and 1
        return 0
        
        
    def sigmoid(self, z):
        #a function to make the sigmoid calculation easier
        return 1/ (1 + np.exp(-z))
    
    
    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
    
    
    def loss(self, X, y):
        #return overall loss (empirical risk) of current weights on X and y
        y_hat = predict(X, self.w)
        return self.logistic_loss(y_hat, y).mean()

        
    def fit(self, X, y, alpha=0.1, max_epochs=1000):
        #no return value, will give instance variable of weights called w, and loss history, and score history
        
        #append ones to the feature matrix
        X = np.append(X, np.ones((X.shape[0],1)),1)
        
        #assign random weight to begin
        p = X.shape[1]
        self.w = np.random.rand(p)
        
        #choose a learning rate, then wt+1 <- wt - alpha gradient descent loss function (wt) until convergence
        for i in range(max_epochs):
            #this is summing up for the gradient of the loss function, but seems like it could be done easier
            sum_derivs = 0
            for i in range(y.size):
                sum_derivs += self.sigmoid(self.predict(X) - y)
                
            #actually get the gradient (i think)
            gradient = (1 / y.size) * sum_derivs
            
            #new weight update the wt+1 <- wt - alpha*gradient
            print("weight " + str(self.w))
            print("alpha times gradient " + str(alpha * gradient))
            new_w = self.w - (alpha * gradient)
            
            self.w = new_w
        

        