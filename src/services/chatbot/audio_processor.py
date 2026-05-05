import numpy as np
import logging
from typing import Dict, Any, Optional
import io
import os
import tempfile
import subprocess

logger = logging.getLogger(__name__)

# Optional librosa for real audio processing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - audio processing will use placeholder values")


def _convert_to_wav(audio_path: str) -> Optional[str]:
    # Convert audio file to WAV via ffmpeg
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "22050", "-ac", "1", tmp.name],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and os.path.getsize(tmp.name) > 0:
            return tmp.name
        os.unlink(tmp.name)
        return None
    except Exception as e:
        logger.warning(f"ffmpeg conversion failed: {e}")
        return None


class AudioProcessor:

    def __init__(self):
        # Standard sample rate for audio analysis
        self.sample_rate = 22050
        self.librosa_available = LIBROSA_AVAILABLE

    def extract_features_from_file(self, audio_path: str) -> Dict[str, float]:
        if not self.librosa_available:
            logger.warning("librosa not available, returning placeholder values")
            return self._get_placeholder_features()

        wav_path = None
        try:
            # Convert to WAV to avoid audioread distortion
            wav_path = _convert_to_wav(audio_path)
            load_path = wav_path if wav_path else audio_path

            y, sr = librosa.load(load_path, sr=self.sample_rate)

            if len(y) == 0 or np.max(np.abs(y)) == 0:
                logger.warning("Audio signal empty or silent, returning placeholders")
                return self._get_placeholder_features()

            # Extract all four audio feature signals
            features = {
                "pause_frequency": self._detect_pauses(y, sr),
                "tremor_intensity": self._detect_tremors(y, sr),
                "speech_rate": self._calculate_speech_rate(y, sr),
                "emotion_intensity": self._detect_emotion_intensity(y, sr)
            }

            logger.info(f"Audio features extracted: {features}")
            return features

        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return self._get_placeholder_features()
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    def extract_features_from_bytes(self, audio_bytes: bytes) -> Dict[str, float]:
        if not self.librosa_available:
            logger.warning("librosa not available, returning placeholder values")
            return self._get_placeholder_features()

        try:
            # Load audio directly from bytes buffer
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=self.sample_rate)

            features = {
                "pause_frequency": self._detect_pauses(y, sr),
                "tremor_intensity": self._detect_tremors(y, sr),
                "speech_rate": self._calculate_speech_rate(y, sr),
                "emotion_intensity": self._detect_emotion_intensity(y, sr)
            }

            return features

        except Exception as e:
            logger.error(f"Error extracting audio features from bytes: {e}")
            return self._get_placeholder_features()

    def _detect_pauses(self, y: np.ndarray, sr: int) -> float:
        # P5: silence gaps normalized by duration
        try:
            # top_db=30 catches meaningful pauses, not every inter-word gap
            intervals = librosa.effects.split(y, top_db=30)

            pause_count = len(intervals) - 1 if len(intervals) > 1 else 0
            total_duration = len(y) / sr

            if total_duration == 0:
                return 0.0

            pause_frequency = pause_count / total_duration

            # Normal: 0.5-1.0 pauses/sec; high concern: >3.0 pauses/sec
            normalized = min(pause_frequency / 3.0, 1.0)

            return round(normalized, 3)

        except Exception as e:
            logger.error(f"Error detecting pauses: {e}")
            return 0.0

    def _detect_tremors(self, y: np.ndarray, sr: int) -> float:
        # P6: pitch variability indicates vocal tremor
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

            # Collect only voiced frames above 80Hz
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 80:
                    pitch_values.append(pitch)

            if len(pitch_values) < 2:
                return 0.0

            pitch_std = np.std(pitch_values)

            # Normal std: 30-80Hz; severe tremor: >150Hz
            tremor_intensity = min(pitch_std / 150.0, 1.0)

            return round(tremor_intensity, 3)

        except Exception as e:
            logger.error(f"Error detecting tremors: {e}")
            return 0.0

    def _calculate_speech_rate(self, y: np.ndarray, sr: int) -> float:
        # P8: onset events estimate syllables then wpm
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

            if len(onset_frames) == 0:
                return 120.0

            duration_seconds = len(y) / sr

            if duration_seconds == 0:
                return 120.0

            syllables_per_second = len(onset_frames) / duration_seconds

            # 1 word ≈ 1.5 syllables in English
            words_per_minute = syllables_per_second * 60 / 1.5
            words_per_minute = max(50.0, min(words_per_minute, 200.0))

            return round(words_per_minute, 1)

        except Exception as e:
            logger.error(f"Error calculating speech rate: {e}")
            return 120.0

    def _detect_emotion_intensity(self, y: np.ndarray, sr: int) -> float:
        # P7: RMS energy variability signals emotional speech
        try:
            rms = librosa.feature.rms(y=y)[0]

            if len(rms) < 2:
                return 0.0

            rms_std = np.std(rms)
            intensity = min(rms_std * 5, 1.0)

            return round(intensity, 3)

        except Exception as e:
            logger.error(f"Error detecting emotion intensity: {e}")
            return 0.0

    def _get_placeholder_features(self) -> Dict[str, float]:
        # Return neutral defaults when audio unavailable
        return {
            "pause_frequency": 0.0,
            "tremor_intensity": 0.0,
            "speech_rate": 120.0,
            "emotion_intensity": 0.0
        }


# Singleton instance
audio_processor = AudioProcessor()
