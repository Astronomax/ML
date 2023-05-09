import numpy as np
from sklearn.base import RegressorMixin


class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1., delta_converged=1e-2, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size
        
        self.W = None
        self.b = None
        
    def fit(self, X, Y):
        self.W = np.zeros(X.shape[1])
        self.b = 0
        n = X.shape[0]
        
        for k in range(self.max_steps):
            p = np.random.permutation(n)
            X = np.array([X[i] for i in p])
            Y = np.array([Y[i] for i in p])
            for i in range(self.batch_size, n + 1, self.batch_size):
                X_batch = X[i-self.batch_size : i]       
                y_batch = Y[i-self.batch_size : i]
                err = X_batch.dot(self.W) + self.b - y_batch
                gradw = 2 * (X_batch.T.dot(err) / self.batch_size + self.regularization * self.W)
                gradb = 2 * np.sum(err) / self.batch_size
                self.W -= self.lr * gradw
                self.b -= self.lr * gradb

    def predict(self, X):
        return X.dot(self.W) + self.b