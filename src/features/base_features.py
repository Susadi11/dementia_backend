"""
Base Feature Extractor Class

This module provides the base class for all feature extractors,
defining the common interface and utility methods.

Author: Research Team
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class FeatureResult:
    """
    Container for feature extraction results.
    
    Attributes:
        features: Dictionary of feature names to values
        feature_type: Type of features (e.g., 'dementia_indicators')
        metadata: Additional metadata about the extraction
    """
    features: Dict[str, Any]
    feature_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    This class defines the interface that all feature extractors must implement
    and provides common utility methods for feature extraction.
    """
    
    def __init__(self):
        """Initialize the base feature extractor."""
        pass
    
    @abstractmethod
    def extract(self, text: str = None, audio_features: Dict = None) -> FeatureResult:
        """
        Extract features from text and/or audio data.
        
        This method must be implemented by all subclasses.
        
        Args:
            text: Text transcript from user
            audio_features: Dictionary with audio analysis results
            
        Returns:
            FeatureResult containing extracted features
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        Get list of feature names this extractor produces.
        
        Returns:
            List of feature names
        """
        pass
    
    def normalize_score(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """
        Normalize a value to 0-1 range.
        
        Args:
            value: Value to normalize
            min_val: Minimum possible value
            max_val: Maximum possible value
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def calculate_average(self, values: List[float]) -> float:
        """
        Calculate average of values, handling empty lists.
        
        Args:
            values: List of values
            
        Returns:
            Average value or 0 if empty
        """
        return sum(values) / len(values) if values else 0.0


__all__ = ["BaseFeatureExtractor", "FeatureResult"]
