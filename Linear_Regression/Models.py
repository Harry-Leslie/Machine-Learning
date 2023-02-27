from Error.functions import mse, mse_prime

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
