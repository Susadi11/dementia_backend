"""
Feature Extractor

Main module for extracting and combining features from both
voice and text analysis for dementia detection.

The 10 parameters extracted are:
1. Semantic incoherence
2. Repeated questions
3. Self-correction
4. Low-confidence answer
5. Hesitation pauses
6. Vocal tremors
7. Emotion + slip
8. Slowed speech
9. Evening errors
10. In-session decline
"""

from .components import VoiceAnalyzer, TextProcessor


class FeatureExtractor:
    """
    Extracts features from conversational data (text and voice)
    for dementia detection.
    """

    def __init__(self):
        """Initialize the Feature Extractor with components."""
        self.voice_analyzer = VoiceAnalyzer()
        self.text_processor = TextProcessor()

    def extract_features(self, audio_data, text_data):
        """
        Extract all features from conversational data.

        Args:
            audio_data: Audio input data
            text_data: Text transcript

        Returns:
            Dictionary containing all 10 extracted parameters
        """
        pass
