"""
Audio Feature Extraction for Dementia Detection

Uses librosa to extract real audio features:
- P5: Hesitation pauses
- P6: Vocal tremors
- P8: Slowed speech
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import io

logger = logging.getLogger(__name__)

# Try to import librosa (optional dependency)
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - audio processing will use placeholder values")


class AudioProcessor:
    """Extract audio features for dementia detection"""

    def __init__(self):
        """Initialize audio processor"""
        self.sample_rate = 22050  # Standard sample rate
        self.librosa_available = LIBROSA_AVAILABLE

    def extract_features_from_file(self, audio_path: str) -> Dict[str, float]:
        """
        Extract audio features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio features
        """
        if not self.librosa_available:
            logger.warning("librosa not available, returning placeholder values")
            return self._get_placeholder_features()

        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Extract features
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

    def extract_features_from_bytes(self, audio_bytes: bytes) -> Dict[str, float]:
        """
        Extract audio features from audio bytes.

        Args:
            audio_bytes: Audio file bytes

        Returns:
            Dictionary with audio features
        """
        if not self.librosa_available:
            logger.warning("librosa not available, returning placeholder values")
            return self._get_placeholder_features()

        try:
            # Load audio from bytes
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=self.sample_rate)

            # Extract features
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
        """
        P5: Detect pause frequency (hesitation pauses).

        Method:
        - Use silence detection to find pauses
        - Count pauses and normalize by duration

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Pause frequency (0-1)
        """
        try:
            # Detect non-silent intervals
            intervals = librosa.effects.split(y, top_db=20)

            # Count pauses (gaps between intervals)
            pause_count = len(intervals) - 1 if len(intervals) > 1 else 0

            # Calculate total duration
            total_duration = len(y) / sr  # seconds

            if total_duration == 0:
                return 0.0

            # Pause frequency = pauses per second
            pause_frequency = pause_count / total_duration

            # Normalize to 0-1 scale
            # Normal: 0-0.2 pauses/sec, High: >0.5 pauses/sec
            # Using 0.5 as max threshold for normalization
            normalized = min(pause_frequency / 0.5, 1.0)

            return round(normalized, 3)

        except Exception as e:
            logger.error(f"Error detecting pauses: {e}")
            return 0.0

    def _detect_tremors(self, y: np.ndarray, sr: int) -> float:
        """
        P6: Detect vocal tremors (voice shakiness).

        Method:
        - Extract pitch using piptrack
        - Calculate pitch variability (jitter)
        - High variability = tremor

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Tremor intensity (0-1)
        """
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

            # Get pitch values where magnitude > 0
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) < 2:
                return 0.0

            # Calculate pitch variability (standard deviation)
            pitch_std = np.std(pitch_values)

            # Normalize tremor intensity
            # Normal std: 0-20 Hz, High std: >50 Hz
            tremor_intensity = min(pitch_std / 50.0, 1.0)

            return round(tremor_intensity, 3)

        except Exception as e:
            logger.error(f"Error detecting tremors: {e}")
            return 0.0

    def _calculate_speech_rate(self, y: np.ndarray, sr: int) -> float:
        """
        P8: Calculate speech rate (words per minute).

        Method:
        - Use onset detection to estimate syllables
        - Approximate words from syllables
        - Calculate words per minute

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Speech rate (words per minute)
        """
        try:
            # Detect onset events (approximate syllables)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

            if len(onset_frames) == 0:
                return 120.0  # Default normal speech rate

            # Calculate duration
            duration_seconds = len(y) / sr

            if duration_seconds == 0:
                return 120.0

            # Estimate syllables per second
            syllables_per_second = len(onset_frames) / duration_seconds

            # Approximate words per minute
            # Average context: 1 word approx 1.5 syllables in English
            words_per_minute = syllables_per_second * 60 / 1.5

            # Clamp to reasonable range
            words_per_minute = max(50.0, min(words_per_minute, 200.0))

            return round(words_per_minute, 1)

        except Exception as e:
            logger.error(f"Error calculating speech rate: {e}")
            return 120.0

    def _detect_emotion_intensity(self, y: np.ndarray, sr: int) -> float:
        """
        P7: Detect emotional intensity in voice.

        Method:
        - Analyze energy (RMS) variability
        - High variability often indicates emotional speech

        Args:
            y: Audio signal
            sr: Sample rate

        Returns:
            Emotion intensity (0-1)
        """
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            if len(rms) < 2:
                return 0.0
                
            # Calculate energy variability
            rms_std = np.std(rms)
            
            # Normalize
            # Heuristic normalization
            intensity = min(rms_std * 5, 1.0)
            
            return round(intensity, 3)
            
        except Exception as e:
            logger.error(f"Error detecting emotion intensity: {e}")
            return 0.0

    def _get_placeholder_features(self) -> Dict[str, float]:
        """Return neutral placeholder features if audio processing fails"""
        return {
            "pause_frequency": 0.0,
            "tremor_intensity": 0.0,
            "speech_rate": 120.0,
            "emotion_intensity": 0.0
        }


# Create singleton instance
audio_processor = AudioProcessor()
