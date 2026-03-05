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
import subprocess
import numpy as np
import tempfile

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in the system or common installation paths."""
    # Try direct PATH first
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check common Windows installation paths
    if os.name == 'nt':  # Windows
        common_paths = [
            r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        for ffmpeg_path in common_paths:
            if os.path.exists(ffmpeg_path):
                try:
                    # Add to PATH for this session
                    ffmpeg_dir = os.path.dirname(ffmpeg_path)
                    if ffmpeg_dir not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"
                    
                    subprocess.run(
                        [ffmpeg_path, "-version"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    logger.info(f"Found FFmpeg at: {ffmpeg_path}")
                    return True
                except Exception:
                    pass
    
    return False


def convert_audio_format(audio_path: str) -> Optional[str]:
    """
    Convert unsupported audio formats to WAV using pydub if available.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Path to WAV file (converted), or None if conversion failed
    """
    # Check if already WAV
    if audio_path.lower().endswith('.wav'):
        return audio_path
    
    # Try using pydub for format conversion
    try:
        from pydub import AudioSegment
        
        logger.info(f"Attempting to convert {audio_path} to WAV format...")
        
        # Load audio with pydub (supports .3gp, .aac, etc. if ffmpeg is installed)
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            logger.warning(f"pydub could not load {audio_path}: {e}")
            return None
        
        # Export as WAV to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            wav_path = temp_wav.name
        
        audio.export(wav_path, format="wav")
        logger.info(f"Successfully converted to WAV: {wav_path}")
        return wav_path
            
    except ImportError:
        logger.debug("pydub not installed for audio format conversion")
        return None
    except Exception as e:
        logger.warning(f"Audio format conversion failed: {e}")
        return None


def load_audio_librosa(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio using librosa (fallback when ffmpeg is not available).
    Supports format conversion using pydub if needed.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (Whisper expects 16kHz)
        
    Returns:
        Audio data as numpy array
        
    Raises:
        Exception: If audio cannot be loaded or converted
    """
    import librosa
    import soundfile as sf
    
    try:
        # Try librosa first (best case)
        logger.info(f"Loading audio with librosa: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        logger.info(f"Successfully loaded audio with librosa")
        return audio
    except Exception as librosa_error:
        logger.warning(f"librosa failed: {librosa_error}")
        
        # Check file extension
        file_ext = audio_path.lower().split('.')[-1]
        logger.info(f"Audio file extension: .{file_ext}")
        
        # Try format conversion for unsupported formats
        if file_ext not in ['wav', 'mp3', 'flac', 'ogg']:
            logger.info(f"Format .{file_ext} requires conversion to WAV")
            converted_path = convert_audio_format(audio_path)
            
            if converted_path and converted_path != audio_path:
                # Retry with converted WAV file
                try:
                    logger.info(f"Retrying with converted audio: {converted_path}")
                    audio, _ = librosa.load(converted_path, sr=sr, mono=True)
                    logger.info(f"Successfully loaded converted audio with librosa")
                    return audio
                except Exception as retry_error:
                    logger.error(f"Failed to load converted audio: {retry_error}")
            else:
                # Conversion failed or not attempted
                logger.error(
                    f"Cannot convert .{file_ext} files without ffmpeg. "
                    f"Please install ffmpeg from https://ffmpeg.org/download.html"
                )
        
        # Try soundfile as fallback for supported formats
        try:
            logger.info("Trying soundfile as fallback")
            audio, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sample_rate != sr:
                import scipy.signal
                audio = scipy.signal.resample(
                    audio,
                    int(len(audio) * sr / sample_rate)
                )
            
            logger.info("Successfully loaded audio with soundfile")
            return audio.astype(np.float32)
            
        except Exception as soundfile_error:
            logger.error(f"soundfile also failed: {soundfile_error}")
            # Re-raise with helpful context
            raise RuntimeError(
                f"Cannot load audio file {audio_path}. "
                f"Formats like .3gp require ffmpeg. "
                f"Install ffmpeg from: https://ffmpeg.org/download.html"
            ) from soundfile_error


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
        self.ffmpeg_available = check_ffmpeg_available()
        
        if not self.ffmpeg_available:
            logger.warning(
                "⚠️ ffmpeg not found in system PATH. "
                "Using librosa/soundfile fallback for audio loading. "
                "For better performance, install ffmpeg: https://ffmpeg.org/download.html"
            )
        else:
            logger.info("✅ ffmpeg detected and available")

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

            logger.info("[SUCCESS] Whisper model loaded successfully!")

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
                    logger.info("[SUCCESS] Whisper model loaded successfully on CPU fallback!")
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

            # Load audio - use librosa if ffmpeg is not available
            if self.ffmpeg_available:
                # Use Whisper's default audio loading (requires ffmpeg)
                audio_input = audio_path
            else:
                # Use librosa fallback
                logger.info("Loading audio with librosa (ffmpeg not available)...")
                audio_input = load_audio_librosa(audio_path)

            # Transcribe
            result = self.model.transcribe(
                audio_input,
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

            logger.info(f"[SUCCESS] Transcription complete: '{transcription_result['text'][:50]}...'")
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
