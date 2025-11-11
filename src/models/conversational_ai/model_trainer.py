"""
Model Trainer for Dementia Detection

Trains machine learning models using conversational AI features
to predict dementia risk.

Author: Research Team
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any
import pickle


class ModelTrainer:
    """
    Trains machine learning models for dementia detection.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'logistic')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        
    def _create_model(self, model_type: str):
        """Create ML model based on type."""
        if model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == 'gradient_boost':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model.
        
        Args:
            X: Feature array
            y: Target labels (0 = No Dementia, 1 = Possible Dementia)
            test_size: Proportion of data to use for testing
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return self.metrics
    
    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Feature array
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = np.array([[1-p, p] for p in predictions])
        
        return predictions, probabilities
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'metrics': self.metrics,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.metrics = data['metrics']
            self.is_trained = data['is_trained']


class ModelRegistry:
    """
    Registry for managing multiple trained models.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self.models = {}
        self.best_model = None
        self.best_f1 = 0
    
    def register_model(self, name: str, model: ModelTrainer):
        """
        Register a trained model.
        
        Args:
            name: Model name
            model: Trained ModelTrainer instance
        """
        self.models[name] = model
        
        # Track best model by F1 score
        if model.metrics.get('f1', 0) > self.best_f1:
            self.best_f1 = model.metrics['f1']
            self.best_model = name
    
    def get_model(self, name: str) -> ModelTrainer:
        """Get a model by name."""
        return self.models.get(name)
    
    def get_best_model(self) -> Tuple[str, ModelTrainer]:
        """Get the best performing model."""
        if self.best_model is None:
            return None, None
        return self.best_model, self.models[self.best_model]
    
    def list_models(self) -> Dict[str, Dict]:
        """List all models with their metrics."""
        return {
            name: model.metrics 
            for name, model in self.models.items()
        }


__all__ = ["ModelTrainer", "ModelRegistry"]
