"""
Data Validator Module

Validates extracted features and chat data.
Ensures data quality before analysis.

Author: Research Team
"""

from typing import Dict, List, Tuple, Optional


class FeatureValidator:
    """Validate extracted features for dementia indicators."""
    
    # List of all required dementia indicator features
    REQUIRED_FEATURES = [
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
    
    @staticmethod
    def validate_features(features: Dict) -> Tuple[bool, List[str]]:
        """
        Validate extracted features dictionary.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check all required features are present
        for feature in FeatureValidator.REQUIRED_FEATURES:
            if feature not in features:
                errors.append(f"Missing feature: {feature}")
            elif not isinstance(features[feature], (int, float)):
                errors.append(f"Invalid type for {feature}: expected number, got {type(features[feature])}")
            elif not (0 <= features[feature] <= 1):
                errors.append(f"Feature {feature} out of range [0, 1]: {features[feature]}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_single_feature(feature_name: str, value: float) -> Tuple[bool, Optional[str]]:
        """
        Validate a single feature value.
        
        Args:
            feature_name: Name of the feature
            value: Feature value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, (int, float)):
            return False, f"Expected number, got {type(value)}"
        
        if not (0 <= value <= 1):
            return False, f"Value {value} out of range [0, 1]"
        
        return True, None
    
    @staticmethod
    def fill_missing_features(features: Dict, default_value: float = 0.0) -> Dict:
        """
        Handle missing features with default values.
        
        Args:
            features: Feature dictionary (may be incomplete)
            default_value: Default value for missing features
            
        Returns:
            Features dict with all required features
        """
        filled_features = features.copy()
        
        for feature in FeatureValidator.REQUIRED_FEATURES:
            if feature not in filled_features:
                filled_features[feature] = default_value
        
        return filled_features


class AudioFeatureValidator:
    """Validate audio features extracted from voice."""
    
    REQUIRED_AUDIO_FEATURES = [
        'pause_frequency',
        'tremor_intensity',
        'emotion_intensity',
        'speech_error_rate',
        'speech_rate'
    ]
    
    @staticmethod
    def validate_audio_features(audio_features: Dict) -> Tuple[bool, List[str]]:
        """
        Validate audio features dictionary.
        
        Args:
            audio_features: Audio analysis results
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for key in AudioFeatureValidator.REQUIRED_AUDIO_FEATURES:
            if key not in audio_features:
                errors.append(f"Missing audio feature: {key}")
            elif not isinstance(audio_features[key], (int, float)):
                errors.append(f"Invalid type for {key}: expected number")
            elif key != 'speech_rate':  # speech_rate has different range
                if not (0 <= audio_features[key] <= 1):
                    errors.append(f"Audio feature {key} out of range [0, 1]")
            else:
                if not (50 <= audio_features[key] <= 200):
                    errors.append(f"Speech rate out of range [50, 200] wpm")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_speech_rate(speech_rate: float) -> Tuple[bool, Optional[str]]:
        """
        Validate speech rate value.
        
        Args:
            speech_rate: Words per minute
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(speech_rate, (int, float)):
            return False, "Speech rate must be a number"
        
        if not (50 <= speech_rate <= 200):
            return False, "Speech rate must be between 50-200 wpm"
        
        return True, None


class ChatMessageValidator:
    """Validate chat messages and sessions."""
    
    @staticmethod
    def validate_message(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a chat message.
        
        Args:
            text: Message text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Message must be text string"
        
        if len(text.strip()) == 0:
            return False, "Message cannot be empty"
        
        if len(text) > 5000:
            return False, "Message too long (max 5000 characters)"
        
        return True, None
    
    @staticmethod
    def validate_session(messages: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate a chat session (multiple messages).
        
        Args:
            messages: List of message texts
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(messages, list):
            return False, "Messages must be a list"
        
        if len(messages) == 0:
            return False, "Session must contain at least one message"
        
        if len(messages) > 100:
            return False, "Session too long (max 100 messages)"
        
        # Validate each message
        for i, msg in enumerate(messages):
            is_valid, error = ChatMessageValidator.validate_message(msg)
            if not is_valid:
                return False, f"Message {i+1}: {error}"
        
        return True, None


__all__ = ["FeatureValidator", "AudioFeatureValidator", "ChatMessageValidator"]
