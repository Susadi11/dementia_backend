"""
Dementia Indicator Analyzer

Integrates NLPEngine (spaCy, NLTK, BERT), VoiceAnalyzer and Whisper-based transcription
to extract text and voice features and compute a simple risk score.
"""

from typing import Dict, Optional
import logging

from .nlp.nlp_engine import NLPEngine
from .components.voice.voice_analyzer import VoiceAnalyzer
from src.preprocessing.voice_processor import get_voice_processor

logger = logging.getLogger(__name__)


class DementiaIndicatorAnalyzer:
    """Top-level analyzer that combines text and audio analysis."""

    def __init__(self, spacy_model: str = "en_core_web_sm", bert_model: str = "bert-base-uncased", device: str = "cpu"):
        self.nlp_engine = NLPEngine(spacy_model=spacy_model, bert_model=bert_model, device=device)
        self.voice_analyzer = VoiceAnalyzer()
        # Whisper processor (for transcription)
        self.voice_processor = get_voice_processor(model_size="base")

    def extract(self, text: Optional[str] = None, audio_path: Optional[str] = None) -> Dict:
        """
        Extract dementia-related indicators from text and/or audio.

        Returns a dict with text_features, voice_features, and risk_score.
        """
        # Step 1: If audio provided and text not provided, transcribe via Whisper
        transcript = text
        if audio_path and not transcript:
            try:
                result = self.voice_processor.transcribe_audio(audio_path)
                if result.get('success'):
                    transcript = result.get('transcript', '')
                else:
                    transcript = ''
            except Exception as e:
                logger.warning(f"Transcription failed: {e}")
                transcript = ''

        # Step 2: Text analysis via NLPEngine
        text_features = {}
        if transcript:
            try:
                text_features = self.nlp_engine.extract_dementia_markers(transcript)
            except Exception as e:
                logger.error(f"Text feature extraction failed: {e}")
                text_features = {}

        # Step 3: Voice analysis
        voice_features = {}
        if audio_path:
            try:
                voice_features = self.voice_analyzer.analyze(audio_path=audio_path)
            except Exception as e:
                logger.error(f"Voice feature extraction failed: {e}")
                voice_features = {}

        # Step 4: Simple risk scoring (weighted sum)
        risk = 0.0
        weights = {
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

        # Combine text and voice features
        for feat, w in weights.items():
            val = 0.0
            if feat in text_features:
                val = float(text_features.get(feat, 0.0))
            elif feat in voice_features:
                val = float(voice_features.get(feat, 0.0))
            risk += val * w

        risk_score = min(1.0, risk)

        return {
            'text_features': text_features,
            'voice_features': voice_features,
            'risk_score': round(risk_score, 3)
        }


__all__ = ['DementiaIndicatorAnalyzer']
