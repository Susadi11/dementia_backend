"""
Text Model Trainer for Dementia Detection

Trains machine learning models specifically for text-based dementia detection.
Uses linguistic features: semantic incoherence, repeated questions, self-corrections, etc.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any, Optional, List
import pickle
import logging

logger = logging.getLogger(__name__)

# Text-specific features
TEXT_FEATURES = [
    'semantic_incoherence',
    'repeated_questions',
    'self_correction',
    'low_confidence_answers',
    'hesitation_pauses',
    'emotion_slip',
    'evening_errors'
]


class TextModelTrainer:
    """
    Trains machine learning models specifically for text-based dementia detection.
    """

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the text model trainer.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'logistic')
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.feature_names = TEXT_FEATURES
        logger.info(f"Initialized TextModelTrainer with model type: {model_type}")

    def _create_model(self, model_type: str):
        """Create ML model based on type."""
        if model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == 'gradient_boost':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42)
        else:
            logger.warning(f"Unknown model type {model_type}, using RandomForest")
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def extract_text_features(self, feature_dict: Dict[str, float]) -> List[float]:
        """
        Extract only text features from a feature dictionary.

        Args:
            feature_dict: Dictionary containing all features

        Returns:
            List of text feature values
        """
        text_features = []
        for feature in TEXT_FEATURES:
            text_features.append(feature_dict.get(feature, 0.0))
        return text_features

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the text model.

        Args:
            X: Feature array (should contain text features)
            y: Target labels (0 = No Dementia, 1 = Possible Dementia)
            test_size: Proportion of data to use for testing

        Returns:
            Dictionary containing model metrics
        """
        logger.info(f"Training text model with {X.shape[0]} samples and {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)

        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0))
        }

        logger.info(f"Training complete. Metrics: {self.metrics}")
        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using text features.

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

    def predict_from_dict(self, feature_dict: Dict[str, float]) -> Tuple[int, float]:
        """
        Make a prediction from a feature dictionary.

        Args:
            feature_dict: Dictionary containing features

        Returns:
            Tuple of (prediction, confidence)
        """
        text_features = self.extract_text_features(feature_dict)
        X = np.array([text_features])

        predictions, probabilities = self.predict(X)
        confidence = float(np.max(probabilities[0]))

        return int(predictions[0]), confidence

    def save_model(self, filepath: str):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'metrics': self.metrics,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }, f)
        logger.info(f"Text model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.metrics = data['metrics']
            self.is_trained = data['is_trained']
            self.feature_names = data.get('feature_names', TEXT_FEATURES)
            self.model_type = data.get('model_type', 'random_forest')
        logger.info(f"Text model loaded from {filepath}")


__all__ = ["TextModelTrainer", "TEXT_FEATURES"]
