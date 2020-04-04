import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import normalize
#Imports pour les scores
from libscores import get_metric
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import preprocessing_function as pf

class model (BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples= 38563
        self.num_feat=59
        self.num_labels=1
        self.is_trained=False
        self.preprocess = PCA(n_components=48)
        self.mod = DecisionTreeRegressor(max_depth = 11, max_features = 33, min_samples_leaf = 29, min_samples_split = 12) 
    
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        if X.ndim>1: self.num_feat = X.shape[1]
        if y.ndim>1: self.num_labels = y.shape[1]
        
        pf.heure_pointe(X)
        outliers = pf.lightFilter(X)
        X = pf.majData(X, outliers)
        X_preprocess = self.preprocess.fit_transform(X)
        X_prprocesse =  sklearn.preprocessing.normalize(X_preprocesse)
        self.mod.fit(X_preprocess, y)
        self.is_trained = True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        y = np.zeros([num_test_samples, self.num_labels])
        X_preprocess = self.preprocess.transform(X)
        y = self.mod.predict(X_preprocess)
        return y

    def save(self, path="./"):
        pass

    def load(self, path="./"):
        pass
