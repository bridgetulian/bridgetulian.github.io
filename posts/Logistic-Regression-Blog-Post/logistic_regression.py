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
        return X@self.w > 0
        
        
    def score(self, X, y):
        #return accuracy of predictions as number between 0 and 1
        y_hat = 1 * (self.predict(X))
        #this returns an array where 1 is they match, 0 is they don't
        corr_pred = 1 * (y_hat==y)
        num_correct = np.sum(corr_pred)
        return num_correct / corr_pred.size
        
        
    def sigmoid(self, z):
        #a function to make the sigmoid calculation easier
        return 1/ (1 + np.exp(-z))
    
    
    def logistic_loss(self, y_hat, y):
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))
    
    
    def loss(self, X, y):
        #return overall loss (empirical risk) of current weights on X and y
        y_hat = X@self.w
        return self.logistic_loss(y_hat, y).mean()
    
    def gradient(self, X, y):
        #easier to get the gradient
        
        n = X.shape[0]
        sum_derivs = 0
        
        for i in range(n):
            X_i = X[i, :]
            y_i = y[i]
            
            sum_derivs += (self.sigmoid(np.dot(self.w, X_i)) - y_i)*X_i
           
        return (1/n) * sum_derivs

        
    def fit(self, X, y, alpha=0.001, max_epochs=1000):
        #no return value, will update instance variable of weights called w, and loss history, and score history
        
        #append ones to the feature matrix
        X = np.append(X, np.ones((X.shape[0],1)),1)
        
        #assign random weight to begin
        p = X.shape[1]
        self.w = np.random.rand(p)
        
        #choose a learning rate, then wt+1 <- wt - alpha gradient descent loss function (wt) until convergence
        for i in range(max_epochs):
            
            gradient = self.gradient(X, y)
            
            score2 = self.score(X,y)
            self.score_history.append(score2)
            
            loss2 = self.loss(X, y)

            self.loss_history.append(loss2)
            
            #new weight update the wt+1 <- wt - alpha*gradient
            new_w = self.w - (alpha * gradient)
            
            self.w = new_w
                
               
    def fit_stochastic(self, X, y, alpha=0.001, m_epochs=1000, batch_size=10):
        #returns nothing, but updates instance variables of weights called w and loss history and score history for stochastic gradient descent
       
        #append ones to the feature matrix
        X = np.append(X, np.ones((X.shape[0],1)),1)
        
        n = X.shape[0]
        
        #set random weight to begin
        p = X.shape[1]
        self.w = np.random.rand(p)
        
        for j in np.arange(m_epochs):

            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]
                grad = self.gradient(x_batch, y_batch) 
                
                score2 = self.score(X,y)
                self.score_history.append(score2)
                
                new_w = self.w - (alpha * grad)
                
                self.w = new_w
            
            loss2 = self.loss(X, y)
            self.loss_history.append(loss2)


   

#stochastic loss pick i from some subset, ideally it isn't too far off from the actual average
#batch size comes in , pick random batch from data, and then you just sample randomly every time
#how do you choose a correct batch size, just give it a number (can just use 1 as a batch set)