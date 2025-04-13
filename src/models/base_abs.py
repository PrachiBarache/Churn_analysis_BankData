# src/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
import joblib
import os

class BaseModel(ABC):
    """Abstract base class for all churn prediction models."""
    
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.feature_importances_ = None
        
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on the provided data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Return probability estimates for samples."""
        pass
    
    def save(self, path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet, cannot save.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load a trained model from disk."""
        self.model = joblib.load(path)
        
    def get_feature_importance(self):
        """Return feature importances if available."""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_[0])
        else:
            raise NotImplementedError("Feature importance not implemented for this model")
        return self.feature_importances_