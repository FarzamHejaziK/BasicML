
import numpy as np
class Precep:

    def __init__(self, learning_rate = 0.01, iter = 100) -> None:
        self.learning_rate =learning_rate
        self.iter = iter
        
    
    def fit(self,X,y):
        self.W = np.random.normal(0,0.1,X.shape[1]+1) 
        self.Loss = []
        for i in range(self.iter):
            y_est = np.heaviside(np.matmul(X,self.W[1:]) + self.W[0],0)
            y_delta = y - y_est
            self.Loss.append(np.sum(y_delta**2)/len(y) + 0.01* np.sum(self.W**2)/len(self.W) )
            self.W[0] += self.learning_rate * np.sum(y_delta)/len(y) + self.learning_rate * 2 * 0.01 * self.W[0] 
            self.W[1:] += self.learning_rate * np.matmul(X.T,y_delta)/len(y) + self.learning_rate * 2 * 0.01 * self.W[1:] 

    def predict(self,X):
        return np.heaviside(np.matmul(X,self.W[1:]) + self.W[0],0)


