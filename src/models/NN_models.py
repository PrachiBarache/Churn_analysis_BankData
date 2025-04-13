import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    """Neural Network for churn prediction."""
    
    def __init__(self, name="neural_network", params=None):
        default_params = {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'batch_size': 32,
            'epochs': 50,
            'patience': 10,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__(name, default_params)
        self.model = None
        
    def _build_model(self, input_dim):
        """Build a neural network model with the specified architecture."""
        tf.random.set_seed(self.params['random_state'])
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.params['hidden_layers'][0], 
                       input_dim=input_dim, 
                       activation=self.params['activation']))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))
        
        # Hidden layers
        for units in self.params['hidden_layers'][1:]:
            model.add(Dense(units, activation=self.params['activation']))
            model.add(BatchNormalization())
            model.add(Dropout(self.params['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation=self.params['output_activation']))
        
        # Compile model
        model.compile(
            optimizer=self.params['optimizer'],
            loss=self.params['loss'],
            metrics=self.params['metrics']
        )
        
        return model
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the neural network model."""
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.params['patience'],
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Use validation data if provided, otherwise use validation split
        validation_data = None
        validation_split = 0.2
        
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
            
        self.model.fit(
            X_train, y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self
        
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict(X).flatten()
        
    def save(self, path):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet, cannot save.")
        self.model.save(path)
        
    def load(self, path):
        """Load a trained model from disk."""
        self.model = load_model(path)
        
    def get_feature_importance(self):
        """
        Calculate feature importance using permutation importance.
        Note: This is a simplified implementation. For production use,
        consider using libraries like eli5 or SHAP for more robust results.
        """
        raise NotImplementedError(
            "Feature importance for neural networks requires specialized libraries like SHAP or ELI5.")