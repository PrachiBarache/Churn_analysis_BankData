import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression for churn prediction."""
    
    def __init__(self, name="logistic_regression", params=None):
        default_params = {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = LogisticRegression(**self.params)
        
    def train(self, X_train, y_train):
        """Train the logistic regression model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class DecisionTreeModel(BaseModel):
    """Decision Tree for churn prediction."""
    
    def __init__(self, name="decision_tree", params=None):
        default_params = {
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = DecisionTreeClassifier(**self.params)
        
    def train(self, X_train, y_train):
        """Train the decision tree model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class KNNModel(BaseModel):
    """K-Nearest Neighbors for churn prediction."""
    
    def __init__(self, name="knn", params=None):
        default_params = {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto',
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = KNeighborsClassifier(**self.params)
        
    def train(self, X_train, y_train):
        """Train the KNN model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)