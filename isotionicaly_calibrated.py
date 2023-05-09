from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd

class IsotonicallyCalibratedModel(ClassifierMixin):
    def __init__(self, base_estimator=None, cv=5):
        self.base_estimator = base_estimator
        self.cv = cv
        self.estimators = []
        self.isotonics = []
        self.estimator = None
        self.isotonic = None
    
    def fit(self, X=None, y=None):
        '''
        X :       numpy.ndarray or pd.DataFrame (n_samples, n_features)
        y :       numpy.ndarray or pd.DataFrame (n_samples,)
        '''
        X = np.array(X)
        y = np.array(y)
        
        
        split = []
        if type(self.cv) is int:
            split = KFold(n_splits=self.cv).split(X)
        else:
            split = self.cv

        for fold in split:
            train_estimator = np.array(fold[0])
            train_isotonic = np.array(fold[1])
            X_estimator = np.array([X[int(i)] for i in train_estimator])
            y_estimator = np.array([y[int(i)] for i in train_estimator])
            X_isotonic = np.array([X[int(i)] for i in train_isotonic])
            y_isotonic = np.array([y[int(i)] for i in train_isotonic])
            estimator = clone(self.base_estimator)
            estimator.fit(X_estimator, y_estimator)
            isotonic = IsotonicRegression(out_of_bounds="clip")       
            isotonic.fit(estimator.predict_proba(X_isotonic).T[1], y_isotonic)
            self.estimators.append(estimator)
            self.isotonics.append(isotonic)
        return self
        
            
    def predict_proba(self, X=None):
        '''
        X :       numpy.ndarray or pd.DataFrame (n_samples, n_features)
        returns : numpy.ndarray (n_samples, 2)
        (column 0 - probability of class 0; column 1 - probability of class 1)
        '''
        n, k = X.shape
        probas = np.zeros(n)
        for i, estimator in enumerate(self.estimators):
            probas += self.isotonics[i].transform(estimator.predict_proba(X).T[1])
        probas /= len(self.estimators)
        return np.array([[1 - p, p] for p in probas])
        
    
    def predict(self, X=None):
        '''
        X :       numpy.ndarray or pd.DataFrame (n_samples, n_features)
        returns : numpy.ndarray (n_samples,)
        '''
        return np.argmax(self.predict_proba(X), axis=1)