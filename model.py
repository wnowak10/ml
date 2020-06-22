import numpy as np
import sklearn.base as sklb
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class custom_skrtrees(sklb.BaseEstimator,sklb.ClassifierMixin):
    def __init__(self, max_depth=None, max_features='auto', n_estimators=500, min_samples_leaf=1, n_jobs=-1, random_state=42):
        self.model = RandomForestClassifier()
        self.max_depth = max_depth
        self.max_features= max_features
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def get_params(self, deep=True):
        return {'max_depth': self.max_depth
               ,'max_features': self.max_features
               ,'n_estimators': self.n_estimators
               ,'min_samples_leaf': self.min_samples_leaf
               , 'n_jobs' : self.n_jobs
               , 'random_state' : self.random_state
               }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def fit(self, X, y):
        self.classes_ = list(set(y))
        self.n_classes = len(self.classes_)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    
    def predict_proba(self,X):
        probas = self.model.predict_proba(X)
        return probas

