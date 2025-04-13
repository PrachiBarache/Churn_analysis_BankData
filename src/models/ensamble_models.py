from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest for churn prediction."""
    
    def __init__(self, name="random_forest", params=None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = RandomForestClassifier(**self.params)
        
    def train(self, X_train, y_train):
        """Train the random forest model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting for churn prediction."""
    
    def __init__(self, name="gradient_boosting", params=None):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = GradientBoostingClassifier(**self.params)
        
    def train(self, X_train, y_train):
        """Train the gradient boosting model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """XGBoost for churn prediction."""
    
    def __init__(self, name="xgboost", params=None):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'scale_pos_weight': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = xgb.XGBClassifier(**self.params)
        
    def train(self, X_train, y_train):
        """Train the XGBoost model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    """LightGBM for churn prediction."""
    
    def __init__(self, name="lightgbm", params=None):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary',
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = lgb.LGBMClassifier(**self.params)
        
    def train(self, X_train, y_train):
        """Train the LightGBM model."""
        self.model.fit(X_train, y_train)
        return self
        
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)