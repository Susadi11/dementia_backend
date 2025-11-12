"""
Voice Processing Module

Handles audio file processing with:
- Automatic Speech Recognition (ASR) using OpenAI Whisper
- Noise filtering and speech enhancement
- Audio preprocessing and validation
- Transcript generation and storage
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
import whisper
from datetime import datetime

try:
    import noisereduce as nr
except ImportError:
    nr = None

from config import config

logger = logging.getLogger(__name__)


class VoiceProcessor:
    """
    Main voice processing class handling audio input and transcription.

    Implements:
    - Audio upload and validation
    - OpenAI Whisper ASR
    - Noise filtering and enhancement
    - Transcript generation and storage
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize voice processor.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                       Default: "base" (good balance between speed and accuracy)
        """
        self.model_size = model_size
        self.sample_rate = config.features.audio_sample_rate  # 16kHz
        self.max_audio_length = config.processing.max_audio_length
        self.supported_formats = config.processing.supported_audio_formats

        # Initialize Whisper model
        logger.info(f"Loading Whisper model ({model_size})...")
        try:
            self.whisper_model = whisper.load_model(model_size)
            logger.info(f"Whisper model {model_size} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

        self.audio_dir = config.paths.data_dir / "audio_uploads"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.transcripts_dir = config.paths.output_dir / "transcripts"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate audio file format and properties."""
        try:
            if not os.path.exists(file_path):
                return False, "Audio file not found"

            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format {file_ext}. Supported: {self.supported_formats}"

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            max_size_mb = config.api.max_upload_size
            if file_size_mb > max_size_mb:
                return False, f"File too large ({file_size_mb:.1f}MB). Max: {max_size_mb}MB"

            y, sr = librosa.load(file_path, sr=None)
            duration_sec = librosa.get_duration(y=y, sr=sr)

            if duration_sec > self.max_audio_length:
                return False, f"Audio too long ({duration_sec:.1f}s). Max: {self.max_audio_length}s"

            if duration_sec < 1:
                return False, "Audio too short (minimum 1 second)"

            return True, f"Valid audio file ({duration_sec:.1f}s)"

        except Exception as e:
            return False, f"Error validating audio: {str(e)}"

    def enhance_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance audio quality using noise reduction and normalization."""
        try:
            if nr is not None:
                logger.info("Applying noise reduction...")
                audio = nr.reduce_noise(y=audio, sr=sr)

            logger.info("Normalizing audio volume...")
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                target_db = -20
                current_db = 20 * np.log10(max_val)
                gain_db = target_db - current_db
                gain_linear = 10 ** (gain_db / 20)
                audio = audio * gain_linear

            audio = np.tanh(audio)

            return audio

        except Exception as e:
            logger.warning(f"Audio enhancement failed: {str(e)}. Using original audio.")
            return audio

    def preprocess_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file (16kHz mono, enhanced, normalized)."""
        logger.info(f"Loading audio from {file_path}...")

        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

        logger.info(f"Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz")

        audio = self.enhance_audio(audio, sr)

        logger.info("Audio preprocessing complete")

        return audio, sr

    def transcribe_audio(
        self,
        file_path: str,
        language: str = "en",
        verbose: bool = False
    ) -> Dict:
        """Transcribe audio using OpenAI Whisper with validation and enhancement."""
        try:
            is_valid, message = self.validate_audio_file(file_path)
            if not is_valid:
                logger.error(f"Audio validation failed: {message}")
                return {
                    'success': False,
                    'transcript': '',
                    'confidence': 0.0,
                    'segments': [],
                    'duration': 0,
                    'language': language,
                    'error': message
                }

            logger.info(f"Transcribing audio: {message}")

            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            audio, sr = self.preprocess_audio(file_path)

            logger.info("Running Whisper ASR...")
            result = self.whisper_model.transcribe(
                file_path,
                language=language,
                verbose=verbose,
                temperature=0.2
            )

            transcript = result.get('text', '').strip()
            segments = result.get('segments', [])
            detected_language = result.get('language', language)

            confidences = []
            for segment in segments:
                if 'confidence' in segment:
                    confidences.append(segment['confidence'])

            avg_confidence = np.mean(confidences) if confidences else 0.9

            logger.info(f"Transcription complete: {len(transcript)} chars, {len(segments)} segments")

            return {
                'success': True,
                'transcript': transcript,
                'confidence': round(float(avg_confidence), 3),
                'segments': segments,
                'duration': round(duration, 2),
                'language': detected_language,
                'error': None
            }

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return {
                'success': False,
                'transcript': '',
                'confidence': 0.0,
                'segments': [],
                'duration': 0,
                'language': language,
                'error': str(e)
            }

    def save_transcript(
        self,
        transcript: str,
        session_id: str,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """Save transcript to file with metadata."""
        try:
            filename = f"{user_id}_{session_id}_{datetime.now().isoformat()}.txt"
            file_path = self.transcripts_dir / filename

            with open(file_path, 'w') as f:
                f.write(f"User ID: {user_id}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                if metadata:
                    f.write(f"Duration: {metadata.get('duration', 0)}s\n")
                    f.write(f"Language: {metadata.get('language', 'unknown')}\n")
                    f.write(f"Confidence: {metadata.get('confidence', 0)}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(transcript)

            logger.info(f"Transcript saved to {file_path}")
            return True, str(file_path)

        except Exception as e:
            logger.error(f"Failed to save transcript: {str(e)}")
            return False, str(e)

    def save_audio(
        self,
        audio: np.ndarray,
        sr: int,
        session_id: str,
        user_id: str,
        original_format: str = "wav"
    ) -> Tuple[bool, str]:
        """Save processed audio to file."""
        try:
            filename = f"{user_id}_{session_id}_{datetime.now().isoformat()}.{original_format}"
            file_path = self.audio_dir / filename

            sf.write(str(file_path), audio, sr)
            logger.info(f"Audio saved to {file_path}")
            return True, str(file_path)

        except Exception as e:
            logger.error(f"Failed to save audio: {str(e)}")
            return False, str(e)

    def process_audio_file(
        self,
        file_path: str,
        user_id: str,
        session_id: str,
        language: str = "en"
    ) -> Dict:
        """Complete pipeline: validate, transcribe, enhance, and store audio."""
        logger.info(f"Starting voice processing for user {user_id}, session {session_id}")

        transcription = self.transcribe_audio(file_path, language=language)

        if not transcription['success']:
            return {
                'success': False,
                'transcript': '',
                'confidence': 0.0,
                'audio_path': '',
                'transcript_path': '',
                'duration': 0,
                'language': language,
                'error': transcription['error']
            }

        try:
            audio, sr = self.preprocess_audio(file_path)
            audio_success, audio_path = self.save_audio(
                audio, sr, session_id, user_id
            )

            transcript_success, transcript_path = self.save_transcript(
                transcription['transcript'],
                session_id,
                user_id,
                metadata={
                    'duration': transcription['duration'],
                    'language': transcription['language'],
                    'confidence': transcription['confidence']
                }
            )

            result = {
                'success': True,
                'transcript': transcription['transcript'],
                'confidence': transcription['confidence'],
                'audio_path': audio_path if audio_success else '',
                'transcript_path': transcript_path if transcript_success else '',
                'duration': transcription['duration'],
                'language': transcription['language'],
                'error': None
            }

            logger.info("Voice processing complete")
            return result

        except Exception as e:
            logger.error(f"Error in voice processing: {str(e)}")
            return {
                'success': False,
                'transcript': transcription['transcript'],
                'confidence': transcription['confidence'],
                'audio_path': '',
                'transcript_path': '',
                'duration': transcription['duration'],
                'language': transcription['language'],
                'error': str(e)
            }


_voice_processor = None


def get_voice_processor(model_size: str = "base") -> VoiceProcessor:
    """Get or create voice processor singleton instance."""
    global _voice_processor
    if _voice_processor is None:
        _voice_processor = VoiceProcessor(model_size=model_size)
    return _voice_processor
