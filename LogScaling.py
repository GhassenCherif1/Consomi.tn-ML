import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
class LogScaling(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self   

    def transform(self, X):
        return np.log1p(X)