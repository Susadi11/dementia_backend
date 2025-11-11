"""
Data Preprocessing Module

Handles cleaning, validation, and feature selection for conversational data.

Author: Research Team
"""

import re
import numpy as np
from typing import Dict, List, Tuple


class TextCleaner:
    """Clean and normalize text data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing noise and normalizing.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters except punctuation
        text = re.sub(r'[^a-z0-9\s\.\?\!,;:\'\"]', '', text)
        
        return text
    
    @staticmethod
    def remove_filler_words(text: str) -> str:
        """
        Remove common filler words.
        
        Args:
            text: Input text
            
        Returns:
            Text with filler words removed
        """
        fillers = ['um', 'uh', 'like', 'you know', 'i mean', 'basically']
        
        for filler in fillers:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', text).strip()


class AudioProcessor:
    """Process audio feature data."""
    
    @staticmethod
    def validate_audio_features(audio_features: Dict) -> bool:
        """
        Validate audio features dictionary.
        
        Args:
            audio_features: Audio analysis results
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = [
            'pause_frequency',
            'tremor_intensity',
            'emotion_intensity',
            'speech_error_rate',
            'speech_rate'
        ]
        
        for key in required_keys:
            if key not in audio_features:
                return False
            if not isinstance(audio_features[key], (int, float)):
                return False
            if not (0 <= audio_features[key] <= 1):
                return False
        
        return True
    
    @staticmethod
    def normalize_audio_features(audio_features: Dict) -> Dict:
        """
        Normalize audio features to 0-1 range.
        
        Args:
            audio_features: Audio analysis results
            
        Returns:
            Normalized audio features
        """
        normalized = {}
        
        for key, value in audio_features.items():
            # Ensure all values are between 0 and 1
            normalized[key] = max(0, min(1, float(value)))
        
        return normalized


class DataValidator:
    """Validate feature data."""
    
    @staticmethod
    def validate_features(features: Dict) -> Tuple[bool, List[str]]:
        """
        Validate extracted features.
        
        Args:
            features: Extracted feature dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        required_features = [
            'semantic_incoherence',
            'repeated_questions',
            'self_correction',
            'low_confidence_answer',
            'hesitation_pauses',
            'vocal_tremors',
            'emotion_slip',
            'slowed_speech',
            'evening_errors',
            'in_session_decline'
        ]
        
        for feature in required_features:
            if feature not in features:
                errors.append(f"Missing feature: {feature}")
            elif not isinstance(features[feature], (int, float)):
                errors.append(f"Invalid type for {feature}: expected number")
            elif not (0 <= features[feature] <= 1):
                errors.append(f"Feature {feature} out of range [0, 1]")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def handle_missing_features(features: Dict, default_value: float = 0.0) -> Dict:
        """
        Handle missing features with default values.
        
        Args:
            features: Feature dictionary
            default_value: Default value for missing features
            
        Returns:
            Features dict with missing values filled
        """
        required_features = [
            'semantic_incoherence',
            'repeated_questions',
            'self_correction',
            'low_confidence_answer',
            'hesitation_pauses',
            'vocal_tremors',
            'emotion_slip',
            'slowed_speech',
            'evening_errors',
            'in_session_decline'
        ]
        
        for feature in required_features:
            if feature not in features:
                features[feature] = default_value
        
        return features


class FeatureScaler:
    """Scale and normalize features."""
    
    def __init__(self):
        """Initialize feature scaler."""
        self.mean = None
        self.std = None
    
    def fit(self, features_array: np.ndarray):
        """
        Fit scaler on features.
        
        Args:
            features_array: 2D array of features
        """
        self.mean = np.mean(features_array, axis=0)
        self.std = np.std(features_array, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
    
    def transform(self, features_array: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            features_array: Features to transform
            
        Returns:
            Scaled features
        """
        if self.mean is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return (features_array - self.mean) / self.std
    
    def fit_transform(self, features_array: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            features_array: Features to fit and transform
            
        Returns:
            Scaled features
        """
        self.fit(features_array)
        return self.transform(features_array)


__all__ = ["TextCleaner", "AudioProcessor", "DataValidator", "FeatureScaler"]
