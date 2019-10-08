# required packages
import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.01):
        """
        this function initialize weight matrix and store alpha in the perceptron
        :param N: number of perceptron units
        :param alpha: learning_rate
        """
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a column of once so that bias can be a trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        #loop over desired number of epochs
        for epoch in np.arange(0, epochs):
            #loop over each individual data entry
            for (x, target) in zip(X, y):
                #take the dot product of feature matrix and weight matrix
                # then pass it to step function
                p = self.step(np.dot(x, self.W))

                #update weight only if prediction does not match target
                if p != target:
                    error = p - target
                    # update weight matrix
                    self.W += -self.alpha * error * x


    def predict(self, X, addBias=True):
        #ensure our input is a matrix
        X = np.atleast_2d(X)

        #check if the bias column should be added
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product of input matrix and weights and pass it to step function
        return self.step(np.dot(X, self.W))

