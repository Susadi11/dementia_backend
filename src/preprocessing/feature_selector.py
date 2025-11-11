"""
Feature Selector Module

Handles feature selection and importance ranking for dementia indicators.

Author: Research Team
"""

from typing import Dict, List, Tuple
import numpy as np


class FeatureSelector:
    """Select and rank important dementia indicator features."""
    
    # Feature weights based on clinical relevance for dementia detection
    FEATURE_WEIGHTS = {
        'semantic_incoherence': 0.12,      # High importance
        'repeated_questions': 0.12,         # High importance
        'self_correction': 0.10,
        'low_confidence_answer': 0.10,
        'hesitation_pauses': 0.10,
        'vocal_tremors': 0.10,
        'emotion_slip': 0.10,
        'slowed_speech': 0.10,
        'evening_errors': 0.08,
        'in_session_decline': 0.08
    }
    
    @staticmethod
    def get_feature_importance(features: Dict) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary mapping features to their importance in the analysis
        """
        importance = {}
        
        for feature_name, value in features.items():
            weight = FeatureSelector.FEATURE_WEIGHTS.get(feature_name, 0.1)
            importance[feature_name] = value * weight
        
        return importance
    
    @staticmethod
    def rank_features(features: Dict) -> List[Tuple[str, float]]:
        """
        Rank features by their contribution to overall risk.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            List of (feature_name, importance_score) sorted by importance
        """
        importance = FeatureSelector.get_feature_importance(features)
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    @staticmethod
    def select_top_features(features: Dict, top_n: int = 5) -> Dict[str, float]:
        """
        Select top N most important features.
        
        Args:
            features: Dictionary of feature values
            top_n: Number of top features to select
            
        Returns:
            Dictionary of top features
        """
        ranked = FeatureSelector.rank_features(features)
        top_features = {name: features[name] for name, _ in ranked[:top_n]}
        return top_features
    
    @staticmethod
    def get_high_indicators(features: Dict, threshold: float = 0.5) -> Dict[str, float]:
        """
        Get features indicating dementia risk (above threshold).
        
        Args:
            features: Dictionary of feature values
            threshold: Threshold above which feature indicates risk
            
        Returns:
            Dictionary of high-risk indicators
        """
        high_indicators = {
            name: value for name, value in features.items()
            if value > threshold
        }
        return high_indicators
    
    @staticmethod
    def filter_features(features: Dict, include_list: List[str] = None) -> Dict[str, float]:
        """
        Filter features to include only specified ones.
        
        Args:
            features: Dictionary of feature values
            include_list: List of feature names to include
            
        Returns:
            Filtered features dictionary
        """
        if include_list is None:
            return features
        
        return {
            name: value for name, value in features.items()
            if name in include_list
        }
    
    @staticmethod
    def normalize_features(features: Dict) -> Dict[str, float]:
        """
        Normalize features to 0-1 range if not already.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Normalized features
        """
        normalized = {}
        
        for name, value in features.items():
            # Ensure all values are between 0 and 1
            normalized[name] = max(0.0, min(1.0, float(value)))
        
        return normalized


class FeatureTransformer:
    """Transform features for better analysis."""
    
    @staticmethod
    def log_transform(value: float) -> float:
        """
        Apply log transformation to a feature value.
        Useful for skewed distributions.
        
        Args:
            value: Feature value
            
        Returns:
            Log-transformed value
        """
        # Add small constant to avoid log(0)
        return np.log(value + 0.01)
    
    @staticmethod
    def square_root_transform(value: float) -> float:
        """
        Apply square root transformation to a feature value.
        
        Args:
            value: Feature value
            
        Returns:
            Square root transformed value
        """
        return np.sqrt(value)
    
    @staticmethod
    def zscore_normalize(features: Dict) -> Dict[str, float]:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Z-score normalized features
        """
        values = np.array(list(features.values()))
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return features
        
        normalized = {}
        for name, value in features.items():
            z_score = (value - mean) / std
            # Map back to 0-1 range
            normalized[name] = max(0.0, min(1.0, (z_score + 3) / 6))
        
        return normalized


__all__ = ["FeatureSelector", "FeatureTransformer"]
