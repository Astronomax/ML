import contextlib
import inspect
import json
import os
import pathlib
import typing as tp
import uuid
import scipy.special as ss
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.tree import DecisionTreeRegressor

class MyBinaryTreeGradientBoostingClassifier:
    """
    *Binary* gradient boosting with trees using
    negative log-likelihood loss with constant learning rate.
    Trees are to predict logits.
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: estimators count
        :param learning_rate: hard learning rate
        :param seed: global seed
        :param kwargs: kwargs of base estimator which is sklearn TreeRegressor
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_logits = None
        self.rng = np.random.default_rng(seed)
        self.base_estimator = DecisionTreeRegressor
        self.base_estimator_kwargs = kwargs
        signature = inspect.signature(self.base_estimator.__init__)
        self.seed_keyword = None
        if 'seed' in signature.parameters:
            self.seed_keyword = 'seed'
        elif 'random_state' in signature.parameters:
            self.seed_keyword = 'random_state'
        self.estimators = []
        self.loss_history = []  # this is to track model learning process

    def create_new_estimator(self, seed):
        kwargs_dict = self.base_estimator_kwargs
        kwargs_dict[self.seed_keyword] = seed
        estimator = self.base_estimator(**kwargs_dict)
        return estimator

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        """
        compute negative log-likelihood for logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions.
        This is used to track model learning process
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return:
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        return -np.array([y * np.log(sigmoid(logits[i]) + MyBinaryTreeGradientBoostingClassifier.eps) 
                          + (1 - y) * np.log(1 - sigmoid(logits[i]) + MyBinaryTreeGradientBoostingClassifier.eps) 
                  for i, y in enumerate(true_labels)]).sum()

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        """
        compute gradient of log-likelihood w.r.t logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return:
        """
        return np.array([y / (1 + np.exp(logits[i])) + (y - 1) / (1 + np.exp(-logits[i])) 
                         for i, y in enumerate(true_labels)])

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        sequentially fit estimators to reduce residual on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        self.loss_history = []
        # only should be fitted on datasets with binary target
        assert (np.unique(y) == np.arange(2)).all()
        # init predictions with mean target (mind that these are logits!)
        self.initial_logits = ss.logit(y.mean())
        # create starting logits
        n_samples = X.shape[0]
        logits = np.array([self.initial_logits] * n_samples)
        # print(y.shape, X.shape, logits.shape)
        # init loss history with starting negative log-likelihood
        self.loss_history.append(self.cross_entropy_loss(y, logits))
        # sequentially fit estimators with random seeds
        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            # add newly created estimator
            self.estimators.append(self.create_new_estimator(seed))
            # compute gradient
            gradient = self.cross_entropy_loss_gradient(y, logits)
            # fit estimator on gradient residual
            self.estimators[-1].fit(X=X, y=-gradient)
            # adjust logits with learning rate
            logits -= self.learning_rate * self.estimators[-1].predict(X=X)
            # append new loss to history
            self.loss_history.append(self.cross_entropy_loss(y, logits))
        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        :param X: [n_samples]
        :return:
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # init logits using precalculated values
        n_samples = X.shape[0]
        logits = np.array([self.initial_logits] * n_samples)
        # sequentially adjust logits with learning rate
        for estimator in self.estimators:
            logits -= self.learning_rate * self.estimators[-1].predict(X=X)
        # don't forget to convert logits to probabilities
        probas = sigmoid(logits)
        return probas

    def predict(
            self,
            X: np.ndarray
    ):
        """
        calculate predictions using predict_proba
        :param X: [n_samples]
        :return:
        """
        predictions = np.vectorize(lambda x: 1 if x > 0.5 else 0)(self.predict_proba(X))
        return predictions