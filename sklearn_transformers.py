from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        super().__init__()
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for c in self.columns_to_drop:
            assert c in X.columns
        transformed_df = X.drop(labels=self.columns_to_drop, axis='columns', inplace=False)
        return transformed_df


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns
        self.offset = 1.0
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        transformed = X.copy()
        for c in self.columns:
            assert c in transformed.columns
            transformed[c] = np.log(transformed[c] + self.offset)
        return transformed
