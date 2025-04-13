import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomerBasicFeatures(BaseEstimator, TransformerMixin):
    """Extract basic features from customer data."""
    
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None
        
    def fit(self, X, y=None):
        """Identify numerical and categorical columns."""
        # Identify numerical and categorical columns
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.numerical_cols = [col for col in X.columns if X[col].dtype in numeric_dtypes]
        self.categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
        return self
        
    def transform(self, X):
        """Transform the data to extract basic features."""
        X_transformed = X.copy()
        
        # Process numerical features
        for col in self.numerical_cols:
            # Handle missing values
            X_transformed[f"{col}_missing"] = X_transformed[col].isna().astype(int)
            X_transformed[col] = X_transformed[col].fillna(X_transformed[col].median())
            
            # Create basic transformations
            if X_transformed[col].min() >= 0:  # Check if non-negative
                X_transformed[f"{col}_log"] = np.log1p(X_transformed[col])
                X_transformed[f"{col}_sqrt"] = np.sqrt(X_transformed[col])
            
        # Process categorical features
        for col in self.categorical_cols:
            # Handle missing values
            X_transformed[col] = X_transformed[col].fillna('Unknown')
            
            # Count encoding
            count_map = X_transformed[col].value_counts().to_dict()
            X_transformed[f"{col}_count"] = X_transformed[col].map(count_map)
            
        return X_transformed