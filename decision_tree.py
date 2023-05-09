import contextlib
import enum
import json
import os
import pathlib
import typing as tp
import uuid

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import mode
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score


class NodeType(enum.Enum):
    REGULAR = 1
    TERMINAL = 2


def gini(y: np.ndarray) -> float:
    K = set(y)
    P = np.array([np.count_nonzero(y == k) / y.size for k in K])
    return np.array(list(map(lambda p: p * (1 - p), P))).sum()


def weighted_impurity(y_left: np.ndarray, y_right: np.ndarray) -> \
        tp.Tuple[float, float, float]:
    left_impurity = gini(y_left)
    right_impurity = gini(y_right)
    weighted_impurity = (y_left.size * left_impurity + y_right.size * right_impurity) / (y_left.size + y_right.size)
    return weighted_impurity, left_impurity, right_impurity


def create_split(feature_values: np.ndarray, threshold: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    left_idx = np.array([i for (i, y) in enumerate(feature_values) if y <= threshold])
    right_idx = np.array([i for (i, y) in enumerate(feature_values) if y > threshold])
    return left_idx, right_idx

mmax_depth = 0
mmin_samples_split = 0

class MyDecisionTreeNode:
    def __init__(
            self,
            meta: 'MyDecisionTreeClassifier',
            depth,
            node_type: NodeType = NodeType.REGULAR,
            predicted_class: tp.Optional[tp.Union[int, str]] = None,
            left_subtree: tp.Optional['MyDecisionTreeNode'] = None,
            right_subtree: tp.Optional['MyDecisionTreeNode'] = None,
            feature_id: int = None,
            threshold: float = None,
            impurity: float = np.inf
    ):
        self._node_type = node_type
        self._meta = meta
        self._depth = depth
        self._predicted_class = predicted_class
        self._class_proba = None
        self._left_subtree = left_subtree
        self._right_subtree = right_subtree
        self._feature_id = feature_id
        self._threshold = threshold
        self._impurity = impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        lowest_impurity = np.inf
        best_feature_id = None
        best_threshold = None
        lowest_left_child_impurity, lowest_right_child_impurity = None, None
        features = self._meta.rng.permutation(X.shape[1])
        for feature in features:
            current_feature_values = X[:, feature]
            thresholds = np.unique(current_feature_values)
            for threshold in thresholds:
                left_idx, right_idx = create_split(current_feature_values, threshold)
                if right_idx.size == 0:
                    continue
                y_left = y[left_idx]
                y_right = y[right_idx]
                current_weighted_impurity, \
                    current_left_impurity, \
                    current_right_impurity = weighted_impurity(y_left, y_right)
                if current_weighted_impurity <= lowest_impurity:
                    lowest_impurity = current_weighted_impurity
                    best_feature_id = feature
                    best_threshold = threshold
                    lowest_left_child_impurity = current_left_impurity
                    lowest_right_child_impurity = current_right_impurity
        return best_feature_id, best_threshold, lowest_left_child_impurity, lowest_right_child_impurity

    def fit(self, X: np.ndarray, y: np.ndarray):
        global mmax_depth
        global mmin_samples_split

        if self._depth == mmax_depth or y.size < mmin_samples_split or (X == X[0]).all():
            self._node_type = NodeType.TERMINAL
            K = set(y)
            max_frequency = np.array([np.count_nonzero(y == k) for k in K]).max()
            most_frequent_classes = [k for k in K if np.count_nonzero(y == k) == max_frequency]
            self._predicted_class = np.random.choice(most_frequent_classes, 1)[0]
            self._class_proba = np.array([np.count_nonzero(y == k) / y.size for k in K])
            return self

        self._feature_id, self._threshold, left_imp, right_imp = self._best_split(X, y)
        left_idx, right_idx = create_split(X[:, self._feature_id], self._threshold)
        self._left_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth + 1,
            impurity=left_imp
        ).fit(X[left_idx], y[left_idx])
        self._right_subtree = MyDecisionTreeNode(
            meta=self._meta,
            depth=self._depth + 1,
            impurity=right_imp
        ).fit(X[right_idx], y[right_idx])
        return self

    def predict(self, x: np.ndarray):
        if self._node_type is NodeType.TERMINAL:
            return self._predicted_class
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict(x)
        else:
            return self._right_subtree.predict(x)

    def predict_proba(self, x: np.ndarray):
        if self._node_type is NodeType.TERMINAL:
            return self._class_proba
        if x[self._feature_id] <= self._threshold:
            return self._left_subtree.predict(x)
        else:
            return self._right_subtree.predict(x)


class MyDecisionTreeClassifier:
    def __init__(
            self,
            max_depth: tp.Optional[int] = None,
            min_samples_split: tp.Optional[int] = 2,
            seed: int = 0
    ):

        self.root = MyDecisionTreeNode(self, 1)
        self._is_trained = False
        #self.max_depth = max_depth or np.inf
        #self.min_samples_split = min_samples_split or 2
        global mmax_depth
        mmax_depth = max_depth or np.inf
        global mmin_samples_split
        mmin_samples_split = min_samples_split or 2
        self.rng = np.random.default_rng(seed)
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_classes = np.unique(y).shape[0]
        self.root.fit(X, y)
        self._is_trained = True
        return self

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('predict call on untrained model')
        else:
            return np.apply_along_axis(self.root.predict, 1, X)

    def predict_proba(self, X: np.ndarray) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError('predict call on untrained model')
        else:
            return np.apply_along_axis(self.root.predict_proba, 1, X)