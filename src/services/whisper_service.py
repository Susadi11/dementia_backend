"""
Whisper Speech-to-Text Service

Local Whisper model for transcribing voice messages to text.
Uses OpenAI Whisper for high-quality speech recognition.
"""

import whisper
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Service for transcribing audio using local Whisper model.

    Features:
    - Local processing (no API calls)
    - Fast and accurate transcription
    - Multiple language support
    - Confidence scoring
    """

    def __init__(self, model_size: str = "small"):
        """
        Initialize Whisper service.

        Args:
            model_size: Whisper model size
                - tiny: Fastest, less accurate (~1GB)
                - base: Balanced (~1GB)
                - small: Good accuracy (~2GB) [RECOMMENDED]
                - medium: Better accuracy (~5GB)
                - large: Best accuracy (~10GB)
        """
        self.model_size = model_size
        self.model = None
        self.device = None

        logger.info(f"Initializing Whisper service with model size: {model_size}")
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        try:
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            logger.info(f"Loading Whisper model on device: {self.device}")

            # Load model
            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )

            logger.info("✅ Whisper model loaded successfully!")

        except Exception as e:
            # Fallback to CPU if MPS/CUDA fails (common with PyTorch updates on Mac)
            if self.device != "cpu":
                logger.warning(f"Failed to load Whisper on {self.device}: {e}")
                logger.info("Retrying with device='cpu'...")
                self.device = "cpu"
                try:
                    self.model = whisper.load_model(
                        self.model_size,
                        device=self.device
                    )
                    logger.info("✅ Whisper model loaded successfully on CPU fallback!")
                    return
                except Exception as cpu_error:
                    logger.error(f"Error loading Whisper on CPU fallback: {cpu_error}")
                    raise cpu_error
            
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)
            language: Optional language code (e.g., 'en', 'es')
                     If None, language will be auto-detected
            task: Either 'transcribe' or 'translate'
                 - transcribe: Convert speech to text in original language
                 - translate: Convert speech to English text

        Returns:
            Dict containing:
                - text: Transcribed text
                - language: Detected/specified language
                - confidence: Average confidence score
                - duration: Audio duration in seconds
                - segments: List of text segments with timestamps
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Transcribing audio: {audio_path}")

            # Transcribe
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=False  # Use FP32 for better compatibility
            )

            # Calculate average confidence from segments
            segments = result.get("segments", [])
            if segments:
                # Whisper doesn't always provide confidence, use probability if available
                avg_confidence = sum(
                    seg.get("avg_logprob", 0) for seg in segments
                ) / len(segments)
                # Convert log probability to approximate confidence (0-1)
                avg_confidence = min(1.0, max(0.0, (avg_confidence + 1.0)))
            else:
                avg_confidence = 0.8  # Default confidence

            # Get duration from segments or estimate
            duration = segments[-1]["end"] if segments else 0.0

            transcription_result = {
                "text": result["text"].strip(),
                "language": result.get("language", language or "en"),
                "confidence": round(avg_confidence, 3),
                "duration": round(duration, 2),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    }
                    for seg in segments
                ]
            }

            logger.info(f"✅ Transcription complete: '{transcription_result['text'][:50]}...'")
            return transcription_result

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio from bytes (useful for uploaded files).

        Args:
            audio_bytes: Audio file bytes
            filename: Original filename (for format detection)
            language: Optional language code

        Returns:
            Transcription result dict
        """
        import tempfile

        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        try:
            # Transcribe from temp file
            result = self.transcribe(temp_path, language=language)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Global singleton instance
_whisper_instance: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """
    Get or create global Whisper service instance.
    Singleton pattern to avoid loading model multiple times.
    """
    global _whisper_instance

    if _whisper_instance is None:
        # Get model size from environment or use default
        model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
        logger.info("Initializing Whisper service for the first time...")
        _whisper_instance = WhisperService(model_size=model_size)

    return _whisper_instance
