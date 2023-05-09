import contextlib
import inspect
import json
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.tree import DecisionTreeRegressor


class MyAdaBoostClassifier:
    """
    Multiclass AdaBoost implementation with SAMME.R algorithm
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            base_estimator: tp.Type[sklearn.base.BaseEstimator],
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: count of estimators
        :param base_estimator: base estimator (practically tree classifier)
        :param seed: global seed
        :param kwargs: keyword arguments of base estimator
        """
        self.n_classes = None
        self.error_history = []  # this is to track model learning process
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = kwargs
        # deduce which keywords are used to set seed for an estimator (sklearn or own tree implementation)
        signature = inspect.signature(self.base_estimator.__init__)
        self.seed_keyword = None
        if 'seed' in signature.parameters:
            self.seed_keyword = 'seed'
        elif 'random_state' in signature.parameters:
            self.seed_keyword = 'random_state'
        self.estimators = []

    def create_new_estimator(
            self,
            seed: int
    ):
        """
        create new base estimator with proper keywords
        and new *unique* seed
        :param seed:
        :return:
        """
        kwargs_dict = self.base_estimator_kwargs
        kwargs_dict[self.seed_keyword] = seed
        estimator = self.base_estimator(**kwargs_dict)
        return estimator

    def get_new_weights(
            self,
            true_labels: np.ndarray,
            predictions: np.ndarray,
            weights: np.ndarray
    ):
        """
        Calculate new weights according to SAMME.R scheme
        :param true_labels: [n_samples]
        :param predictions: [n_samples, n_classes]
        :param weights:     [n_samples]
        :return: normalized weights for next estimator fitting
        """
        K = predictions.shape[1]
        new_weights = []
        for i, c in enumerate(true_labels):
            p = np.array([max(p, self.eps) for p in predictions[i]])
            y = np.array([-1 / (K - 1) if j != true_labels[i] else 1 for j in range(K)]).T
            new_weights.append(weights[i] * np.exp((1 - K) * y.dot(np.log(p)) / K))
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()
        return new_weights

    @staticmethod
    def get_estimator_error(
            estimator: sklearn.base.BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray
    ):
        """
        calculate weighted error of an estimator
        :param estimator:
        :param X:       [n_samples, n_features]
        :param y:       [n_samples]
        :param weights: [n_samples]
        :return:
        """
        (n_samples, n_features) = X.shape
        error = 0
        weights_sum = weights.sum()
        predict = estimator.predict(X=X)
        for i in range(n_samples):
            if predict[i] != y[i]:
                error += weights[i]
        error /= weights_sum
        return error

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        sequentially fit estimators with updated weights on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        n_samples = len(y)
        self.error_history = []
        # compute number of classes for internal use
        self.n_classes = np.unique(y)
        # init weights uniformly over all samples
        weights = np.array([1 / n_samples] * n_samples)
        # sequentially fit each model and adjust weights

        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            # add newly created estimator
            self.estimators.append(self.create_new_estimator(seed))
            # fit added estimator to data with current sample weights
            self.estimators[-1].fit(X=X, y=y, sample_weight=weights)
            # compute probability predictions
            predictions = self.estimators[-1].predict_proba(X=X)
            # calculate weighted error of last estimator and append to error history
            self.error_history.append(self.get_estimator_error(estimator=self.estimators[-1], X=X, y=y, weights=weights))
            # compute new adjusted weights
            weights = self.get_new_weights(true_labels=y, predictions=predictions, weights=weights)

        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        predicts probability of each class
        :param X: [n_samples, n_features]
        :return: array of probabilities of a shape [n_samples, n_classes]
        """
        # calculate probabilities from each estimator and average them, clip logarithms using self.eps
        #log_p = np.log(np.vectorize(lambda e: e.predict(X=X))(np.array(self.estimators)))
        log_p = np.log(np.maximum(np.array([e.predict_proba(X=X) for e in self.estimators]), self.eps))
        log_p = np.swapaxes(log_p, axis1=0, axis2=1)        
        (n, m, K) = log_p.shape
        probas = []
        for i in range(n):
            cur_log_p = np.array(log_p[i])
            means = np.apply_along_axis(np.mean, axis=1, arr=cur_log_p)
            h = np.array([np.vectorize(lambda lp: (K - 1) * (lp - means[i]))(row) for i, row in enumerate(cur_log_p)])
            # use softmax to ensure probabilities sum to 1, use numerically stable implementation
            probas.append(ss.softmax(np.apply_along_axis(lambda x: x.sum() / (K - 1), axis=0, arr=h)))
        return np.array(probas)

    def predict(
            self,
            X: np.ndarray
    ):
        """
        predicts class (use predicted class probabilities)
        :param X: [n_samples, n_features]
        :return: array class predictions of a shape [n_samples]
        """
        predictions = np.argmax(self.predict_proba(X=X), axis=1)
        return predictions