#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:52:18 2020

@author: marion.pobelle
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

class ModelRegressor(BaseEstimator):
    def __init__(self, n_estimators=10):
        self.n_estimators
        # Preprocessing
        self.scaler = StandardScaler()
        # Regressor
        self.reg = DecisionTreeRegressor(max_depth = 11, max_features = 33, min_samples_leaf = 29, min_samples_split = 12)

    def fit(self, X, y, sample_weights=None):
        X = self.scaler.fit_transform(X)
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.reg.predict(X)
        return y_pred