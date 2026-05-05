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
    # Check ffmpeg in PATH or common install locations
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
    if os.name == 'nt':
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
    # Validate WAV file has actual speech content
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
    # Use ffprobe to get source file duration
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
    # Convert unsupported formats to WAV via pydub
    if audio_path.lower().endswith('.wav'):
        return audio_path

    try:
        from pydub import AudioSegment

        logger.info(f"Attempting to convert {audio_path} to WAV format...")

        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            logger.warning(f"pydub could not load {audio_path}: {e}")
            return None

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
    # Load audio with librosa; convert format if needed
    import librosa
    import soundfile as sf

    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        logger.info(f"Successfully loaded audio with librosa")
        return audio
    except Exception as librosa_error:
        logger.warning(f"librosa failed: {librosa_error}")

        file_ext = audio_path.lower().split('.')[-1]

        # Convert unsupported format then retry
        if file_ext not in ['wav', 'mp3', 'flac', 'ogg']:
            logger.info(f"Format .{file_ext} requires conversion to WAV")
            converted_path = convert_audio_format(audio_path)

            if converted_path and converted_path != audio_path:
                try:
                    audio, _ = librosa.load(converted_path, sr=sr, mono=True)
                    logger.info(f"Successfully loaded converted audio with librosa")
                    return audio
                except Exception as retry_error:
                    logger.error(f"Failed to load converted audio: {retry_error}")
            else:
                logger.error(
                    f"Cannot convert .{file_ext} files without ffmpeg. "
                    f"Please install ffmpeg from https://ffmpeg.org/download.html"
                )

        # soundfile fallback for supported formats
        try:
            logger.info("Trying soundfile as fallback")
            audio, sample_rate = sf.read(audio_path)

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

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
            raise RuntimeError(
                f"Cannot load audio file {audio_path}. "
                f"Formats like .3gp require ffmpeg. "
                f"Install ffmpeg from: https://ffmpeg.org/download.html"
            ) from soundfile_error


# Known Whisper hallucination output patterns
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
    # Detect known Whisper hallucination outputs
    stripped = text.strip().strip(".")
    if not stripped:
        return True
    if text.strip() in _HALLUCINATION_PATTERNS:
        return True
    if all(c in ". \t\n" for c in text):
        return True
    words = stripped.split()
    if len(words) <= 1 and len(stripped) <= 4:
        return True
    return False


class WhisperService:

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None
        self.device = None
        self.ffmpeg_available = check_ffmpeg_available()

        if not self.ffmpeg_available:
            logger.warning(
                "ffmpeg not found in system PATH. "
                "Using librosa/soundfile fallback for audio loading."
            )
        else:
            logger.info("[Whisper] ffmpeg available")

        logger.info(f"[Whisper] Initializing service…")
        logger.info(f"[Whisper] Loading '{model_size}' model…")
        self._load_model()

    def _load_model(self):
        # Auto-detect device; force CPU for Whisper MPS
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            # MPS causes issues with Whisper; use CPU
            if self.device == "mps":
                logger.info("[Whisper] MPS detected but using CPU for Whisper compatibility")
                self.device = "cpu"

            logger.info(f"Loading Whisper model on device: {self.device}")

            self.model = whisper.load_model(
                self.model_size,
                device=self.device
            )

            logger.info("[SUCCESS] Whisper model loaded successfully!")

        except Exception as e:
            # Retry on CPU if MPS or CUDA fails
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
        # Convert mobile audio formats to 16kHz mono WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            converted_wav_path = tmp.name

        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-i", audio_path,
            "-vn",
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
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = os.path.getsize(audio_path)
        logger.info(f"[Whisper] Input file: {audio_path} ({file_size} bytes)")

        converted_wav_path = None
        try:
            # Step 1: Probe source audio duration
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

            # Step 2: Convert to WAV and validate
            if self.ffmpeg_available:
                converted_wav_path = self._convert_to_wav(audio_path)

                if converted_wav_path:
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

            # Step 3: Multi-pass transcription with hallucination retry
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
                return self.model.transcribe(audio_input, **opts)

            # Pass 1: auto-detect language, no bias, temp=0
            result = _do_transcribe(language, temp=0.0)
            raw_text = result["text"].strip()
            logger.info(f"[Whisper] Pass 1 raw='{raw_text}'")

            # Pass 2: retry with English if hallucination detected
            if _is_hallucination(raw_text):
                logger.warning(
                    f"[Whisper] Pass 1 looks like hallucination ('{raw_text}'). "
                    "Retrying with lang=en, temperature=0.2..."
                )
                result = _do_transcribe("en", temp=0.2)
                raw_text = result["text"].strip()
                logger.info(f"[Whisper] Pass 2 raw='{raw_text}'")

            # Pass 3: retry with full temperature schedule
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

            # Return empty if all passes produced hallucinations
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

            # Step 4: Build final result with confidence score
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
        # Save bytes to temp file then transcribe
        import tempfile

        logger.info(
            f"[Whisper] transcribe_from_bytes: {len(audio_bytes)} bytes, "
            f"filename={filename}"
        )

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
    global _whisper_instance
    # Create singleton on first call
    if _whisper_instance is None:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
        logger.info("Initializing Whisper service for the first time...")
        _whisper_instance = WhisperService(model_size=model_size)
    return _whisper_instance
