"""
Generate Sample Audio Files for Dementia Detection Testing

This script generates synthetic audio files with different characteristics
to simulate healthy controls and dementia risk participants.

Usage:
    python data/generate_sample_audio.py

Note: Requires numpy and soundfile to be installed
"""

from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install numpy soundfile librosa")
    DEPS_AVAILABLE = False


def generate_speech_audio(
    duration=10,
    sample_rate=16000,
    pitch_variation=0.0,
    speech_rate_factor=1.0,
    tremor_intensity=0.0,
    pause_frequency=0.0
):
    """
    Generate synthetic speech-like audio with various characteristics.

    Args:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        pitch_variation: Amount of pitch variation (0-1)
        speech_rate_factor: Speech rate multiplier (lower = slower)
        tremor_intensity: Vocal tremor intensity (0-1)
        pause_frequency: Frequency of pauses (0-1)

    Returns:
        numpy array of audio samples
    """
    num_samples = int(duration * sample_rate)
    t = np.arange(num_samples) / sample_rate

    # Base frequency for speech (around 150-250 Hz)
    base_freq = 200 + 50 * np.sin(2 * np.pi * 0.5 * t)

    # Add pitch variation
    freq = base_freq + pitch_variation * 50 * np.sin(2 * np.pi * 2 * t)

    # Generate signal
    signal = np.sin(2 * np.pi * freq * t)

    # Add tremor (amplitude modulation at ~5 Hz)
    if tremor_intensity > 0:
        tremor = tremor_intensity * 0.3 * np.sin(2 * np.pi * 5 * t)
        signal = signal * (1 + tremor)

    # Add pauses (silence periods)
    if pause_frequency > 0:
        pause_length = int(0.5 * sample_rate)  # 0.5 second pauses
        pause_interval = int(duration * sample_rate / (pause_frequency * 5))
        for i in range(0, num_samples, pause_interval):
            signal[i:i + pause_length] *= 0.1

    # Add background noise
    noise = 0.01 * np.random.normal(0, 1, num_samples)
    signal = signal + noise

    # Normalize
    signal = signal / np.max(np.abs(signal))

    # Apply speech rate factor (time stretching simulation)
    if speech_rate_factor < 1.0:
        # Stretch the signal (slower speech)
        stretch_factor = 1.0 / speech_rate_factor
        new_length = int(len(signal) * stretch_factor)
        if new_length > num_samples:
            new_signal = np.zeros(num_samples)
            indices = np.linspace(0, len(signal) - 1, num_samples)
            new_signal = np.interp(indices, np.arange(len(signal)), signal)
            signal = new_signal
        else:
            indices = np.linspace(0, len(signal) - 1, new_length)
            signal = np.interp(indices, np.arange(len(signal)), signal)

    return signal[:num_samples]


def main():
    """Generate sample audio files."""
    if not DEPS_AVAILABLE:
        print("\n❌ Cannot generate sample audio files without dependencies")
        print("Please install required packages:")
        print("  pip install numpy soundfile librosa")
        return

    output_dir = Path(__file__).parent / "sample" / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000

    print("Generating sample audio files...")

    # Sample 1: Control participant (normal, healthy)
    print("  Generating sample_001.wav (Control - Healthy)")
    audio_1 = generate_speech_audio(
        duration=30,
        sample_rate=sample_rate,
        pitch_variation=0.3,
        speech_rate_factor=1.0,
        tremor_intensity=0.05,
        pause_frequency=0.1
    )
    sf.write(output_dir / "sample_001.wav", audio_1, sample_rate)

    # Sample 2: Dementia risk (high hesitation, tremor, slow speech)
    print("  Generating sample_002.wav (Dementia Risk - High Risk)")
    audio_2 = generate_speech_audio(
        duration=30,
        sample_rate=sample_rate,
        pitch_variation=0.6,
        speech_rate_factor=0.7,
        tremor_intensity=0.4,
        pause_frequency=0.6
    )
    sf.write(output_dir / "sample_002.wav", audio_2, sample_rate)

    # Sample 3: Control participant (older, but healthy)
    print("  Generating sample_003.wav (Control - Healthy Older)")
    audio_3 = generate_speech_audio(
        duration=30,
        sample_rate=sample_rate,
        pitch_variation=0.2,
        speech_rate_factor=0.95,
        tremor_intensity=0.08,
        pause_frequency=0.15
    )
    sf.write(output_dir / "sample_003.wav", audio_3, sample_rate)

    # Sample 4: Dementia risk (severe cognitive decline)
    print("  Generating sample_004.wav (Dementia Risk - Severe)")
    audio_4 = generate_speech_audio(
        duration=30,
        sample_rate=sample_rate,
        pitch_variation=0.7,
        speech_rate_factor=0.6,
        tremor_intensity=0.5,
        pause_frequency=0.75
    )
    sf.write(output_dir / "sample_004.wav", audio_4, sample_rate)

    # Sample 5: Control participant (younger)
    print("  Generating sample_005.wav (Control - Younger)")
    audio_5 = generate_speech_audio(
        duration=30,
        sample_rate=sample_rate,
        pitch_variation=0.4,
        speech_rate_factor=1.05,
        tremor_intensity=0.02,
        pause_frequency=0.08
    )
    sf.write(output_dir / "sample_005.wav", audio_5, sample_rate)

    print("\n✅ Sample audio files generated successfully!")
    print(f"   Location: {output_dir}")
    print(f"   Sample rate: {sample_rate} Hz")


if __name__ == "__main__":
    main()
