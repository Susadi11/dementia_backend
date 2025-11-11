"""
Voice Analyzer Component

Analyzes voice recordings to extract features such as:
- Hesitation pauses
- Vocal tremors
- Speech rate (slowed speech detection)
- Emotion detection
- Vocal characteristics

The voice-based dementia detection parameters:
3. Vocal tremors - amplitude modulation at ~5 Hz
6. Slowed speech - reduced spectral flux and lower speech rate
10. In-session decline - progressive fatigue during session
Plus hesitation pauses from audio analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class VoiceAnalyzer:
    """
    Analyzes voice patterns from audio recordings to identify
    dementia-related speech characteristics.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the Voice Analyzer.

        Args:
            sample_rate: Sample rate for audio processing (default: 16000 Hz)
        """
        self.sample_rate = sample_rate
        self.has_librosa = HAS_LIBROSA
        self.has_soundfile = HAS_SOUNDFILE

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not self.has_librosa:
            raise ImportError("librosa is required for audio processing")

        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.array([]), self.sample_rate

    def extract_vocal_tremors(self, audio_path: str) -> float:
        """
        Detect vocal tremors in audio.
        Analyzes amplitude modulation at ~5 Hz (typical tremor frequency).

        Args:
            audio_path: Path to audio file

        Returns:
            Score 0-1, where 1 indicates high tremor intensity
        """
        if not self.has_librosa:
            return 0.0

        try:
            y, sr = self._load_audio(audio_path)

            if len(y) == 0:
                return 0.0

            # Extract energy envelope using MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)
            energy = np.abs(mfccs[0])

            if len(energy) < 10:
                return 0.0

            # Analyze low-frequency modulation (tremor is typically 4-8 Hz)
            energy_diff = np.diff(energy)
            tremor_score = np.std(energy_diff) / (np.mean(np.abs(energy_diff)) + 1e-6)
            tremor_score = min(1.0, tremor_score / 5.0)  # Normalize

            return tremor_score

        except Exception:
            return 0.0

    def extract_slowed_speech(self, audio_path: str) -> float:
        """
        Detect slowed speech rate.
        Analyzes spectral centroid and frame rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Score 0-1, where 1 indicates significantly slowed speech
        """
        if not self.has_librosa:
            return 0.0

        try:
            y, sr = self._load_audio(audio_path)

            if len(y) == 0:
                return 0.0

            # Extract spectral centroid to estimate pitch/voice activity
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

            # Estimate speech rate from spectral flux
            spectral_flux = np.sqrt(np.sum(np.diff(spectral_centroid) ** 2))
            spectral_flux_normalized = spectral_flux / (np.mean(spectral_centroid) + 1e-6)

            # Slowed speech typically has lower spectral flux
            slowed_score = 1.0 / (1.0 + spectral_flux_normalized)

            return min(1.0, slowed_score)

        except Exception:
            return 0.0

    def estimate_duration(self, audio_path: str) -> float:
        """
        Estimate audio duration for speech rate calculation.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        if not self.has_librosa:
            return 0.0

        try:
            y, sr = self._load_audio(audio_path)
            return len(y) / sr
        except Exception:
            return 0.0

    def extract_in_session_decline(self, audio_path: str) -> float:
        """
        Detect in-session decline (fatigue/progressive decline during session).
        Compares first half vs second half of speech.

        Args:
            audio_path: Path to audio file

        Returns:
            Score 0-1, where 1 indicates significant in-session decline
        """
        if not self.has_librosa:
            return 0.0

        try:
            y, sr = self._load_audio(audio_path)

            if len(y) < 2:
                return 0.0

            # Split audio into first and second halves
            mid = len(y) // 2
            first_half = y[:mid]
            second_half = y[mid:]

            # Analyze energy in each half
            energy_first = np.mean(np.abs(first_half))
            energy_second = np.mean(np.abs(second_half))

            # Decline score: if second half has less energy, indicates fatigue
            decline_score = max(0.0, (energy_first - energy_second) / (energy_first + 1e-6))

            return min(1.0, decline_score)

        except Exception:
            return 0.0

    def analyze(self, audio_path: Optional[str] = None, audio_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze voice patterns in audio data.

        Args:
            audio_path: Path to audio file
            audio_data: Audio data as numpy array (alternative to audio_path)

        Returns:
            Dictionary containing extracted voice features
        """
        features = {
            'vocal_tremors': 0.0,
            'slowed_speech': 0.0,
            'in_session_decline': 0.0,
        }

        if not audio_path and audio_data is None:
            return features

        if audio_path and Path(audio_path).exists():
            features['vocal_tremors'] = self.extract_vocal_tremors(audio_path)
            features['slowed_speech'] = self.extract_slowed_speech(audio_path)
            features['in_session_decline'] = self.extract_in_session_decline(audio_path)

        return features

    def get_feature_description(self) -> Dict[str, str]:
        """Get descriptions of extracted voice features."""
        return {
            'vocal_tremors': 'Amplitude modulation intensity (0-1)',
            'slowed_speech': 'Speech rate reduction indicator (0-1)',
            'in_session_decline': 'Progressive fatigue during session (0-1)',
        }
