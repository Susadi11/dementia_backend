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


def validate_audio_wav(wav_path: str, min_duration: float = 0.3, min_rms: float = 0.0005) -> Dict[str, Any]:
    """
    Validate a WAV file has actual speech content.
    
    Returns:
        Dict with 'valid', 'duration', 'rms_energy', 'reason'
    """
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path)
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        duration = len(data) / sr
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
        
        result = {
            'valid': True,
            'duration': round(duration, 2),
            'rms_energy': round(rms, 6),
            'peak': round(peak, 4),
            'samples': len(data),
            'sample_rate': sr,
            'reason': 'ok',
        }
        
        if duration < min_duration:
            result['valid'] = False
            result['reason'] = f'Audio too short: {duration:.2f}s (min {min_duration}s)'
        elif rms < min_rms and peak < 0.005:
            result['valid'] = False
            result['reason'] = f'Audio is silence/near-silence: RMS={rms:.6f}, peak={peak:.4f}'
        
        return result
    except Exception as e:
        return {
            'valid': False,
            'duration': 0,
            'rms_energy': 0,
            'peak': 0,
            'samples': 0,
            'sample_rate': 0,
            'reason': f'Cannot read WAV: {e}',
        }


def probe_audio_duration(audio_path: str) -> Optional[float]:
    """Use ffprobe to get the duration of the source audio file."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


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


# Expanded set of known Whisper hallucination patterns
_HALLUCINATION_PATTERNS = {
    "...", "....", ".....", "......",
    "[ Music ]", "[Music]", "(Music)",
    "[ Silence ]", "[Silence]",
    "you", "You", "you.", "You.",
    "Thank you.", "Thanks for watching.",
    "Thanks for watching!", "Thank you for watching.",
    "Bye.", "Bye!", "Bye bye.",
    "Subtitles by the Amara.org community",
    "ご視聴ありがとうございました",
    "Sous-titres réalisés pour la communauté d'Amara.org",
}


def _is_hallucination(text: str) -> bool:
    """Check if transcription text is a known Whisper hallucination."""
    stripped = text.strip().strip(".")
    if not stripped:
        return True
    if text.strip() in _HALLUCINATION_PATTERNS:
        return True
    if all(c in ". \t\n" for c in text):
        return True
    # Single word with very short text is suspicious
    words = stripped.split()
    if len(words) <= 1 and len(stripped) <= 4:
        return True
    return False


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
            logger.info("[Whisper] ffmpeg available")

        logger.info(f"[Whisper] Initializing service…")
        logger.info(f"[Whisper] Loading '{model_size}' model…")
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        try:
            # Auto-detect device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            # Whisper works best on CPU or CUDA; MPS can cause issues
            # Force CPU for reliability unless CUDA is available
            if self.device == "mps":
                logger.info("[Whisper] MPS detected but using CPU for Whisper compatibility")
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
                    logger.info("[Whisper] Model loaded on CPU fallback")
                    return
                except Exception as cpu_error:
                    logger.error(f"Error loading Whisper on CPU fallback: {cpu_error}")
                    raise cpu_error
            
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def _convert_to_wav(self, audio_path: str) -> Optional[str]:
        """
        Convert audio to 16kHz mono WAV using ffmpeg with robust flags
        for mobile formats (.m4a, .3gp, .aac, .amr).
        
        Returns path to converted WAV, or None on failure.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            converted_wav_path = tmp.name

        # Use more robust ffmpeg flags:
        # -nostdin: prevent ffmpeg from reading stdin (avoids hangs)
        # -vn: skip video streams (some .m4a have cover art)
        # -acodec pcm_s16le: explicit output codec
        # -ar 16000 -ac 1: 16kHz mono as Whisper expects
        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", audio_path,
            "-vn",                    # ignore video/album art
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            converted_wav_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"[Whisper] Converted → {converted_wav_path}")
                return converted_wav_path
            else:
                stderr_tail = result.stderr.decode(errors="replace")[-500:]
                logger.warning(
                    f"[Whisper] ffmpeg conversion failed (rc={result.returncode}): "
                    f"{stderr_tail}"
                )
                # Clean up failed output
                if os.path.exists(converted_wav_path):
                    os.remove(converted_wav_path)
                return None
        except subprocess.TimeoutExpired:
            logger.error("[Whisper] ffmpeg conversion timed out")
            if os.path.exists(converted_wav_path):
                os.remove(converted_wav_path)
            return None
        except Exception as e:
            logger.error(f"[Whisper] ffmpeg conversion error: {e}")
            if os.path.exists(converted_wav_path):
                os.remove(converted_wav_path)
            return None

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

        file_size = os.path.getsize(audio_path)
        logger.info(f"[Whisper] Input file: {audio_path} ({file_size} bytes)")

        converted_wav_path = None
        try:
            # --- Step 1: Probe source duration ---
            source_duration = probe_audio_duration(audio_path) if self.ffmpeg_available else None
            if source_duration is not None:
                logger.info(f"[Whisper] Source audio duration: {source_duration:.2f}s")
                if source_duration < 0.3:
                    logger.warning(f"[Whisper] Audio too short ({source_duration:.2f}s), likely no speech")
                    return {
                        "text": "",
                        "language": language or "en",
                        "confidence": 0.0,
                        "duration": round(source_duration, 2),
                        "segments": [],
                        "warning": "Audio too short to contain speech",
                    }

            # --- Step 2: Convert to WAV ---
            if self.ffmpeg_available:
                converted_wav_path = self._convert_to_wav(audio_path)
                
                if converted_wav_path:
                    # Validate the converted WAV
                    validation = validate_audio_wav(converted_wav_path)
                    logger.info(
                        f"[Whisper] WAV validation: duration={validation['duration']}s, "
                        f"RMS={validation['rms_energy']}, peak={validation['peak']}, "
                        f"valid={validation['valid']}, reason={validation['reason']}"
                    )
                    
                    if not validation['valid']:
                        logger.warning(f"[Whisper] Converted WAV failed validation: {validation['reason']}")
                        return {
                            "text": "",
                            "language": language or "en",
                            "confidence": 0.0,
                            "duration": validation['duration'],
                            "segments": [],
                            "warning": f"Audio validation failed: {validation['reason']}",
                        }
                    
                    audio_input = converted_wav_path
                else:
                    logger.warning("[Whisper] ffmpeg conversion failed, trying original file")
                    audio_input = audio_path
            else:
                logger.info("Loading audio with librosa (ffmpeg not available)...")
                audio_input = load_audio_librosa(audio_path)

            # --- Step 3: Transcribe with multi-pass strategy ---
            def _do_transcribe(lang, temp=0.0, prompt=None):
                opts = dict(
                    language=lang,
                    task=task,
                    fp16=False,
                    condition_on_previous_text=False,
                    temperature=temp,
                )
                if prompt:
                    opts["initial_prompt"] = prompt
                # Do NOT set initial_prompt by default — it biases
                # Whisper toward hallucinating those words on quiet audio
                return self.model.transcribe(audio_input, **opts)

            # Pass 1: auto-detect language, no prompt bias, temp=0
            result = _do_transcribe(language, temp=0.0)
            raw_text = result["text"].strip()
            logger.info(f"[Whisper] Pass 1 raw='{raw_text}'")

            # Pass 2: if hallucination detected, retry with English + slight temperature
            if _is_hallucination(raw_text):
                logger.warning(
                    f"[Whisper] Pass 1 looks like hallucination ('{raw_text}'). "
                    "Retrying with lang=en, temperature=0.2..."
                )
                result = _do_transcribe("en", temp=0.2)
                raw_text = result["text"].strip()
                logger.info(f"[Whisper] Pass 2 raw='{raw_text}'")

            # Pass 3: if still hallucination, try with temperature schedule
            if _is_hallucination(raw_text):
                logger.warning(
                    f"[Whisper] Pass 2 still hallucination ('{raw_text}'). "
                    "Retrying with temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)..."
                )
                result = self.model.transcribe(
                    audio_input,
                    language="en",
                    task=task,
                    fp16=False,
                    condition_on_previous_text=False,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                )
                raw_text = result["text"].strip()
                logger.info(f"[Whisper] Pass 3 raw='{raw_text}'")

            # Final check: if all passes produced hallucinations, return empty
            final_text = result["text"].strip()
            if _is_hallucination(final_text):
                logger.warning(
                    f"[Whisper] All passes produced hallucination: '{final_text}'. "
                    "Returning empty transcription."
                )
                segments = result.get("segments", [])
                duration = segments[-1]["end"] if segments else (source_duration or 0.0)
                return {
                    "text": "",
                    "language": result.get("language", language or "en"),
                    "confidence": 0.0,
                    "duration": round(duration, 2),
                    "segments": [],
                    "warning": f"Could not transcribe speech (Whisper returned: '{final_text}')",
                }

            # --- Step 4: Build result ---
            segments = result.get("segments", [])
            if segments:
                avg_confidence = sum(
                    seg.get("avg_logprob", 0) for seg in segments
                ) / len(segments)
                avg_confidence = min(1.0, max(0.0, (avg_confidence + 1.0)))
            else:
                avg_confidence = 0.8

            duration = segments[-1]["end"] if segments else (source_duration or 0.0)

            transcription_result = {
                "text": final_text,
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

            logger.info(
                f"[Whisper] FINAL text='{transcription_result['text'][:80]}' "
                f"conf={transcription_result['confidence']}"
            )
            return transcription_result

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            raise
        finally:
            if converted_wav_path and os.path.exists(converted_wav_path):
                try:
                    os.remove(converted_wav_path)
                except OSError:
                    pass

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

        logger.info(
            f"[Whisper] transcribe_from_bytes: {len(audio_bytes)} bytes, "
            f"filename={filename}"
        )

        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        try:
            result = self.transcribe(temp_path, language=language)
            return result
        finally:
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
