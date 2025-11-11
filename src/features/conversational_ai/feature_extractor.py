"""
Feature Extractor

Main module for extracting and combining features from both
voice and text analysis for dementia detection.

The 10 parameters extracted are:
1. Semantic incoherence - illogical or off-topic speech
2. Repeated questions - asking same question multiple times
3. Self-correction - instances of self-correcting
4. Low-confidence answers - hesitant or unsure responses
5. Hesitation pauses - filled pauses (um, uh, er)
6. Vocal tremors - amplitude modulation in voice
7. Emotion + slip - inappropriate emotional expressions
8. Slowed speech - reduced speech rate
9. Evening errors - time-dependent cognitive decline
10. In-session decline - progressive fatigue during session
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
from .components import VoiceAnalyzer, TextProcessor


class FeatureExtractor:
    """
    Extracts features from conversational data (text and voice)
    for dementia detection.
    """

    # Feature names mapping to expected outputs
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

    def __init__(self):
        """Initialize the Feature Extractor with components."""
        self.voice_analyzer = VoiceAnalyzer()
        self.text_processor = TextProcessor()

    def extract_features(
        self,
        transcript_text: Optional[str] = None,
        transcript_path: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract all 10 features from conversational data.

        Args:
            transcript_text: Transcript text content
            transcript_path: Path to transcript file
            audio_path: Path to audio file

        Returns:
            Dictionary containing all 10 extracted parameters
        """
        features = {}

        # Load transcript if path provided instead of text
        transcript = transcript_text
        if transcript is None and transcript_path:
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            except Exception as e:
                print(f"Error reading transcript: {e}")
                transcript = None

        # Extract text-based features
        if transcript:
            text_features = self.text_processor.process(transcript)
            features.update(text_features)
        else:
            # Default values if no transcript
            features['semantic_incoherence'] = 0.0
            features['repeated_questions'] = 0.0
            features['self_correction'] = 0.0
            features['low_confidence_answers'] = 0.0
            features['hesitation_pauses'] = 0.0
            features['emotion_slip'] = 0.0
            features['evening_errors'] = 0.0

        # Extract voice-based features
        if audio_path and Path(audio_path).exists():
            voice_features = self.voice_analyzer.analyze(audio_path=audio_path)
            features.update(voice_features)
        else:
            # Default values if no audio
            features['vocal_tremors'] = 0.0
            features['slowed_speech'] = 0.0
            features['in_session_decline'] = 0.0

        return features

    def extract_features_normalized(
        self,
        transcript_text: Optional[str] = None,
        transcript_path: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract features and normalize all values to 0-1 range.

        Args:
            transcript_text: Transcript text content
            transcript_path: Path to transcript file
            audio_path: Path to audio file

        Returns:
            Dictionary with normalized feature values (all 0-1)
        """
        features = self.extract_features(
            transcript_text=transcript_text,
            transcript_path=transcript_path,
            audio_path=audio_path
        )

        # Normalize count-based features to 0-1
        # repeated_questions: normalize count to 0-1 (assume max 10 questions)
        if features.get('repeated_questions', 0) > 0:
            features['repeated_questions'] = min(1.0, features['repeated_questions'] / 10.0)

        # self_correction: normalize count to 0-1 (assume max 10 corrections)
        if features.get('self_correction', 0) > 0:
            features['self_correction'] = min(1.0, features['self_correction'] / 10.0)

        # hesitation_pauses: normalize count to 0-1 (assume max 20 pauses)
        if features.get('hesitation_pauses', 0) > 0:
            features['hesitation_pauses'] = min(1.0, features['hesitation_pauses'] / 20.0)

        return features

    def get_feature_report(
        self,
        features: Dict[str, float]
    ) -> str:
        """
        Generate a human-readable report of extracted features.

        Args:
            features: Dictionary of extracted features

        Returns:
            Formatted report string
        """
        report = "\n" + "="*60 + "\n"
        report += "DEMENTIA DETECTION FEATURE REPORT\n"
        report += "="*60 + "\n\n"

        # Text-based features
        report += "TEXT-BASED FEATURES:\n"
        report += f"  • Semantic Incoherence:    {features.get('semantic_incoherence', 0.0):.3f}\n"
        report += f"  • Repeated Questions:      {features.get('repeated_questions', 0.0):.3f}\n"
        report += f"  • Self-Correction:         {features.get('self_correction', 0.0):.3f}\n"
        report += f"  • Low-Confidence Answers:  {features.get('low_confidence_answers', 0.0):.3f}\n"
        report += f"  • Hesitation Pauses:       {features.get('hesitation_pauses', 0.0):.3f}\n"
        report += f"  • Emotion + Slip:          {features.get('emotion_slip', 0.0):.3f}\n"
        report += f"  • Evening Errors:          {features.get('evening_errors', 0.0):.3f}\n"

        # Voice-based features
        report += "\nVOICE-BASED FEATURES:\n"
        report += f"  • Vocal Tremors:           {features.get('vocal_tremors', 0.0):.3f}\n"
        report += f"  • Slowed Speech:           {features.get('slowed_speech', 0.0):.3f}\n"
        report += f"  • In-Session Decline:      {features.get('in_session_decline', 0.0):.3f}\n"

        report += "\n" + "="*60 + "\n"

        return report

    def get_component_info(self) -> Dict[str, Dict]:
        """
        Get information about feature extractors and their descriptions.

        Returns:
            Dictionary with component info and feature descriptions
        """
        return {
            'text_processor': {
                'description': 'Extracts linguistic features from transcripts',
                'features': self.text_processor.get_feature_description()
            },
            'voice_analyzer': {
                'description': 'Analyzes voice patterns from audio recordings',
                'features': self.voice_analyzer.get_feature_description()
            },
            'all_features': self.FEATURE_NAMES
        }
