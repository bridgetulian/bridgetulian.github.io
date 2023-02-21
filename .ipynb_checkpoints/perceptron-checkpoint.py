import numpy as np
import pandas as pd

class Perceptron:

#method fit 
#method predict
#method score

    def __init__(self):
        self.w = (0,0)
        self.history = []


    def fit(X, y, max_loops=1000):
        #primary method with no return value
        #after p.fit(X,y) the perceptron p has an instance variable of weights called w (this is vector w tilda = (w, -b)
        #it should also have instance variable p.history, list of evolution of the score over the training period
        X_mod = np.append(X, np.ones((X.shape[0], 1)), 1)
        #y = [0,1,1,1,1,0]
        y_mod = y[y == 0] = -1
        print(y_mod)
        self.w = np.random.rand(X.len)
        for i in range(max_steps):
            #perform perceptron update and log score in self.history
            x = 3

    def predict(X):
        #return a vector y hat of predicted labels, model's predictions for the labels on the data
        y = (0,0)
        return y

    def score(X,y):
        #return accuracy of perceptron as a number between 0 and 1, 1 is perfect classification
        acc=0.5
        return acc
    
p = Perceptron()
p.fit([1,3], [0,1,1,1,1,0])