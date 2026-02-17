"""
Chatbot Services Module

Contains all services related to the dementia care chatbot:
- chatbot_service: LLaMA-based conversational AI
- scoring_engine: 12 behavioral parameter scoring
- session_finalizer: Session lifecycle management
- risk_calculator: Weekly risk calculations
- audio_processor: Audio feature extraction
- whisper_service: Speech-to-text transcription
"""

from .chatbot_service import DementiaChatbot, get_chatbot
from .scoring_engine import ScoringEngine
from .session_finalizer import SessionFinalizer, session_finalizer
from .risk_calculator import WeeklyRiskCalculator
from .audio_processor import AudioProcessor, audio_processor
from .whisper_service import WhisperService, get_whisper_service

__all__ = [
    "DementiaChatbot",
    "get_chatbot",
    "ScoringEngine",
    "SessionFinalizer",
    "session_finalizer",
    "WeeklyRiskCalculator",
    "AudioProcessor",
    "audio_processor",
    "WhisperService",
    "get_whisper_service",
]
