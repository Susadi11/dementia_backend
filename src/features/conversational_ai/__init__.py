"""
Conversational AI Feature Module

This module handles the extraction and analysis of features from
user conversations (text and voice) to identify dementia indicators.

Features analyzed:
- Semantic incoherence
- Repeated questions
- Self-correction patterns
- Low-confidence answers
- Hesitation pauses
- Vocal tremors
- Emotion and slip patterns
- Slowed speech
- Evening errors
- In-session decline
"""

from .feature_extractor import FeatureExtractor
from .components.voice import VoiceAnalyzer
from .components.text import TextProcessor

__all__ = ["FeatureExtractor", "VoiceAnalyzer", "TextProcessor"]
