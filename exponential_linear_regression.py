import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import RegressorMixin

class ExponentialLinearRegression(RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.ridge = Ridge(*args, **kwargs)

    def fit(self, X, Y):
        self.ridge = self.ridge.fit(X, np.log(Y))
        return self
    
    def predict(self, X):
        return np.exp(self.ridge.predict(X))

    def get_params(self, *args, **kwargs):
        return self.ridge.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        self.ridge = self.ridge.set_params(*args, **kwargs)
        return self