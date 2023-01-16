import numpy as np

class SVM:

    def __init__(self, learning_rate = 0.01, iter = 100, reg = 0.001 ) -> None:
        self.learning_rate  = learning_rate
        self.iter           = iter
        self.reg            = reg

    def fit(self,X,y):
        self.Loss = []
        self.W = np.random.normal(0,1,(X.shape[1]+1))
        for i in range(self.iter):
            Yc = np.matmul(X,self.W[1:].T) + self.W[0] 
            loss = np.maximum(0,1-np.multiply(Yc,y)) 
            out_select = np.heaviside(loss,0)
            self.Loss.append(np.sum(loss))
            temp = np.multiply(out_select,y)
            self.W[0] += self.learning_rate * np.sum(temp)/len(y) - self.reg * 2 * self.W[0]
            temp2 = np.matmul(temp,X)
            self.W[1:] += self.learning_rate * temp2//len(y) - self.reg * 2 * self.W[1:]
            

    def predict(self,X):
        print(np.matmul(X,self.W[1:].T) + self.W[0])
        return np.sign(np.matmul(X,self.W[1:].T) + self.W[0])



        