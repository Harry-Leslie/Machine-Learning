from Error.functions import mse, mse_prime
from util.functions import commuteMLL

import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.m = None
        self.c = None

    def train(self, X, y, epochs = 15000, learning_rate=0.03, verbose=True, error_graph=False):
        n = len(y)
        ErrorVals = []
        self.X = X
        self.y = y
        
        # Initial Gradient and Intercept of the line before training
        self.m = 1
        self.c = 1
        
        for _ in range(epochs):
            Error = 0
            # calculates the sum of the error
            for i in range(n):
                actual = self.y[i]
                predicted = (self.m*self.X[i])+self.c
                Error+=mse(actual, predicted)
            if verbose:
                if error_graph:
                    ErrorVals.append(Error/n)
                print(Error/n)
            
            # calculating the derivative of the Error with respect to the gradient and intercept
            dE_dm = 0
            dE_dc = 0
            for i in range(n):
                actual = self.y[i]
                predicted = (self.m*self.X[i])+self.c
                dE_dm+=mse_prime(actual, predicted)*self.X[i]
                dE_dc+=mse_prime(actual, predicted)
            dE_dm/=n
            dE_dc/=n

            # applying gradient descent
            self.m = self.m-(learning_rate*dE_dm)
            self.c = self.c-(learning_rate*dE_dc)
        if error_graph:
            plt.plot([i+1 for i in range(len(ErrorVals))], ErrorVals)
            plt.ylabel("Error")
            plt.show()

    def predict(self, x):
        print(self.m * x + self.c)
        return self.m * x + self.c

class Multilinear_Regression:
    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.m = None
        self.c = None
    
    def train(self, X, y, epochs = 15000, learning_rate=0.03, verbose=True, error_graph=False):
        n = len(y)
        ErrorVals = []
        self.X = X
        self.y = y
        
        # Initial Gradient and Intercept of the line before training
        self.m = [1 for i in range(len(self.X[0]))]
        self.c = 1

        
        for _ in range(epochs):
            Error = 0
            for i in range(len(self.X)):
                actual = self.y[i]
                predicted = commuteMLL(self.m, self.X[i], self.c)
                Error+=mse(actual, predicted)
            
            if error_graph:
                ErrorVals.append(Error/n)
            if verbose:
                print(Error/n)
            
            # Vector of Derivatives of Error function with respect to each gradient in each dimension
            dE_dm = [0 for i in range(len(self.X[0]))]
            # bias value
            dE_dc = 0

            for i in range(len(self.X)):
                for j in range(len(dE_dm)):
                    dE_dm[j]+=mse_prime(self.y[i], commuteMLL(self.m, self.X[i], self.c))*self.X[i][j]
                dE_dc+=mse_prime(self.y[i], commuteMLL(self.m, self.X[i], self.c))
            
            for i in range(len(self.m)):
                self.m[i]=self.m[i]-(learning_rate*(dE_dm[i]/n))
            self.c = self.c-(learning_rate*(dE_dc/n))
                
        print(self.m, self.c)

    def predict(self, X):
        commuteMLL(self.m, X, self.c)