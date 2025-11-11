"""
Dementia Detection Model - Predictor

Predicts dementia risk based on extracted conversational features.
Uses a rule-based approach with weighted feature importance.
"""

from typing import Dict, Tuple
import joblib
from pathlib import Path


class DementiaPredictor:
    """
    Dementia risk predictor using ensemble approach.
    Rules-based for sample data, ML-based when real data available.
    """

    # Feature names in the order they should be used
    FEATURE_NAMES = [
        'semantic_incoherence',
        'repeated_questions',
        'self_correction',
        'low_confidence_answers',
        'hesitation_pauses',
        'vocal_tremors',
        'emotion_slip',
        'slowed_speech',
        'evening_errors',
        'in_session_decline'
    ]

    # Feature importance weights based on literature
    FEATURE_WEIGHTS = {
        'semantic_incoherence': 0.12,
        'repeated_questions': 0.14,
        'self_correction': 0.05,
        'low_confidence_answers': 0.10,
        'hesitation_pauses': 0.11,
        'vocal_tremors': 0.08,
        'emotion_slip': 0.07,
        'slowed_speech': 0.12,
        'evening_errors': 0.10,
        'in_session_decline': 0.11,
    }

    def __init__(self, model_dir: str = "./models"):
        """
        Initialize the predictor.

        Args:
            model_dir: Directory to store/load trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.is_trained = False

    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict]:
        """
        Predict dementia risk from features using rule-based approach.

        Args:
            features: Dictionary with 10 feature values

        Returns:
            Tuple of (prediction, risk_score, feature_contributions)
        """
        # Calculate weighted sum
        risk_score = 0.0
        feature_contributions = {}

        for feature_name in self.FEATURE_NAMES:
            feature_value = features.get(feature_name, 0.0)
            weight = self.FEATURE_WEIGHTS[feature_name]

            risk_score += feature_value * weight
            feature_contributions[feature_name] = feature_value * weight

        # Normalize to 0-1
        risk_score = min(1.0, risk_score)

        # Threshold-based classification
        prediction = "dementia_risk" if risk_score > 0.4 else "control"

        return prediction, risk_score, feature_contributions

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance weights.

        Returns:
            Dictionary with feature importance scores
        """
        return self.FEATURE_WEIGHTS.copy()

    def get_model_info(self) -> Dict:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'Rule-based Weighted Predictor',
            'is_trained': self.is_trained,
            'n_features': len(self.FEATURE_NAMES),
            'feature_names': self.FEATURE_NAMES,
            'feature_weights': self.FEATURE_WEIGHTS,
        }


class ModelUtils:
    """
    Utility class for managing machine learning models
    used in dementia detection.
    """

    @staticmethod
    def load_model(model_path):
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the trained model file

        Returns:
            Loaded model object
        """
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    @staticmethod
    def predict(model, features):
        """
        Make predictions using a trained model.

        Args:
            model: Trained model object
            features: Extracted features for prediction

        Returns:
            Prediction result
        """
        if model is None:
            return None
        try:
            return model.predict(features)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
