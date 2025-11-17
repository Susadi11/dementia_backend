"""
Feature Extractor

Main module for extracting and combining features from both voice and text analysis for dementia detection.
Extracts 10+ dementia indicator parameters from conversational data.

Enhanced with NLP analysis using BERT embeddings and advanced linguistic features.
"""

import logging
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from .components.voice import VoiceAnalyzer
from .components.text import TextProcessor

try:
    from .nlp import NLPEngine
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    NLPEngine = None

logger = logging.getLogger(__name__)


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

    def __init__(self, use_nlp: bool = True):
        """
        Initialize the Feature Extractor with components.

        Args:
            use_nlp: Whether to enable NLP analysis (default: True)
        """
        self.voice_analyzer = VoiceAnalyzer()
        self.text_processor = TextProcessor()
        self.nlp_engine = None
        self.use_nlp = use_nlp and NLP_AVAILABLE

        # Initialize NLP engine if available and enabled
        if self.use_nlp:
            try:
                self.nlp_engine = NLPEngine()
                logger.info("NLPEngine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NLPEngine: {e}")
                self.use_nlp = False
                self.nlp_engine = None

    def extract_features(
        self,
        transcript_text: Optional[str] = None,
        transcript_path: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract all 10+ features from conversational data.

        Includes both traditional features and advanced NLP-based features if available.

        Args:
            transcript_text: Transcript text content
            transcript_path: Path to transcript file
            audio_path: Path to audio file

        Returns:
            Dictionary containing all extracted parameters (10+ features)
        """
        features = {}

        transcript = transcript_text
        if transcript is None and transcript_path:
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            except Exception as e:
                logger.error(f"Error reading transcript: {e}")
                transcript = None

        if transcript:
            # Extract traditional text features
            text_features = self.text_processor.process(transcript)
            features.update(text_features)

            # Extract NLP-based features if available
            if self.use_nlp and self.nlp_engine:
                try:
                    nlp_features = self._extract_nlp_features(transcript)
                    features.update(nlp_features)
                except Exception as e:
                    logger.error(f"Error extracting NLP features: {e}")
        else:
            # Default values for missing transcript
            features['semantic_incoherence'] = 0.0
            features['repeated_questions'] = 0.0
            features['self_correction'] = 0.0
            features['low_confidence_answers'] = 0.0
            features['hesitation_pauses'] = 0.0
            features['emotion_slip'] = 0.0
            features['evening_errors'] = 0.0

        if audio_path and Path(audio_path).exists():
            voice_features = self.voice_analyzer.analyze(audio_path=audio_path)
            features.update(voice_features)
        else:
            features['vocal_tremors'] = 0.0
            features['slowed_speech'] = 0.0
            features['in_session_decline'] = 0.0

        return features

    def _extract_nlp_features(self, transcript: str) -> Dict[str, float]:
        """
        Extract advanced NLP-based features using the NLPEngine.

        Args:
            transcript: Transcript text

        Returns:
            Dictionary of NLP-based dementia markers
        """
        if not self.nlp_engine:
            return {}

        try:
            # Perform NLP analysis
            analysis_result = self.nlp_engine.analyze(transcript, include_embeddings=False)

            # Extract dementia markers
            dementia_markers = self.nlp_engine.extract_dementia_markers(analysis_result)

            # Extract speech quality metrics
            speech_quality = self.nlp_engine.extract_speech_quality_metrics(analysis_result)

            # Combine all NLP features with 'nlp_' prefix to distinguish from traditional features
            nlp_features = {}
            for key, value in dementia_markers.items():
                nlp_features[f'nlp_{key}'] = float(value)

            # Add key speech quality metrics
            nlp_features['nlp_semantic_coherence'] = float(speech_quality.get('semantic_coherence', 0.5))
            nlp_features['nlp_discourse_coherence'] = float(speech_quality.get('discourse_coherence', 0.5))
            nlp_features['nlp_lexical_diversity'] = float(speech_quality.get('lexical_diversity', 0.5))

            return nlp_features
        except Exception as e:
            logger.error(f"Error in _extract_nlp_features: {e}")
            return {}

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

        if features.get('repeated_questions', 0) > 0:
            features['repeated_questions'] = min(1.0, features['repeated_questions'] / 10.0)

        if features.get('self_correction', 0) > 0:
            features['self_correction'] = min(1.0, features['self_correction'] / 10.0)

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

        report += "TEXT-BASED FEATURES:\n"
        report += f"  • Semantic Incoherence:    {features.get('semantic_incoherence', 0.0):.3f}\n"
        report += f"  • Repeated Questions:      {features.get('repeated_questions', 0.0):.3f}\n"
        report += f"  • Self-Correction:         {features.get('self_correction', 0.0):.3f}\n"
        report += f"  • Low-Confidence Answers:  {features.get('low_confidence_answers', 0.0):.3f}\n"
        report += f"  • Hesitation Pauses:       {features.get('hesitation_pauses', 0.0):.3f}\n"
        report += f"  • Emotion + Slip:          {features.get('emotion_slip', 0.0):.3f}\n"
        report += f"  • Evening Errors:          {features.get('evening_errors', 0.0):.3f}\n"

        report += "\nVOICE-BASED FEATURES:\n"
        report += f"  • Vocal Tremors:           {features.get('vocal_tremors', 0.0):.3f}\n"
        report += f"  • Slowed Speech:           {features.get('slowed_speech', 0.0):.3f}\n"
        report += f"  • In-Session Decline:      {features.get('in_session_decline', 0.0):.3f}\n"

        report += "\n" + "="*60 + "\n"

        return report

    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about feature extractors and their descriptions.

        Returns:
            Dictionary with component info and feature descriptions
        """
        info = {
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

        # Add NLP engine info if available
        if self.use_nlp and self.nlp_engine:
            info['nlp_engine'] = {
                'description': 'Advanced NLP analysis with BERT embeddings and linguistic features',
                'enabled': True,
                'components': [
                    'Text Preprocessing (normalization, tokenization, lemmatization)',
                    'Semantic Analysis (BERT embeddings, coherence detection)',
                    'Sentiment & Emotion Analysis (sentiment, emotions, confidence)',
                    'Discourse & Linguistic Analysis (discourse markers, syntax complexity)',
                ],
                'dementia_markers': [
                    'semantic_incoherence',
                    'semantic_drift',
                    'repetition_score',
                    'hesitation_score',
                    'low_confidence_score',
                    'emotional_drift',
                    'reduced_complexity',
                    'word_finding_difficulty',
                    'tangentiality',
                    'circumlocution',
                ]
            }

        return info
