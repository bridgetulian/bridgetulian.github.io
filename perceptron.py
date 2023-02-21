import numpy as np
import pandas as pd

class Perceptron:

    def __init__(self):
        self.w = []
        self.history = []              
            
    def predict(self, X):
        #return a vector y hat of predicted labels, model's predictions for the labels on the data
        #matrix multiplication
        return X@self.w > 0

    def score(self, X, y):
        #return accuracy of perceptron as a number between 0 and 1, 1 is perfect classification
        y_hat = 1 * (self.predict(X))
        y_hat[y_hat == 0] = -1
        #this returns an array where 1 is they match, 0 is they don't
        corr_pred = 1 * (y_hat==y)
        num_correct = np.sum(corr_pred)
        return num_correct / corr_pred.size
    
    
    def fit(self, X, y, max_steps=1000):
        #primary method with no return value
        #after p.fit(X,y) the perceptron p has an instance variable of weights called w (this is vector w tilda = (w, -b)
        
        #append ones onto X
        X_mod = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        #this is replacing any 0 labels in the array with -1 for perceptron ease
        y[y == 0] = -1
        
        #assign random weight to begin
        p = X_mod.shape[1]
        self.w = np.random.rand(p)
        
        #for loop over the maximum steps
        for i in range(max_steps):
            
            score2 = self.score(X_mod,y)
         #   print("The accuracy of iteration %d" %i + " is " + str(score2) + ".")
            self.history.append(score2)
            
            #perform perceptron update and log score in self.history
            index = np.random.randint(y.size - 1)
            
            #get the label for that data point
            yi = y[index]

            #getting the ith col (bc i think each col is a new data point?)
            #is a row or a column a new data point
            xi = X_mod[index,:]
            
            #updating the weight
            new_w = self.w + 1*(yi * np.dot(self.w, xi) < 0)*(yi * xi)
            
            #actually updating weight by setting it to w
            self.w = new_w

            
