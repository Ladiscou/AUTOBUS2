#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:12:18 2020

@author: marion.pobelle
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

class ModelRegressor(BaseEstimator):
    def __init__(self, n_estimators=10):
        self.n_estimators
        # Preprocessing
        self.scaler = StandardScaler()
        # Regressor
        self.reg = KNeighborsRegressor(weights = 'uniform', algorithm = 'auto', leaf_size = 42, n_neighbors = 6, p = 2)

    def fit(self, X, y, sample_weights=None):
        X = self.scaler.fit_transform(X)
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.reg.predict(X)
        return y_pred