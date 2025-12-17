"""
Voice Data Generator for Reminder System

Generates synthetic voice data and integrates with text data for
comprehensive multi-modal training of the reminder system.

Uses text-to-speech and audio processing to create realistic
voice samples with dementia-related characteristics.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import json
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import random
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import voice processing components
try:
    from src.preprocessing.voice_processor import VoiceProcessor
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Voice processing not available, using simulated features")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceReminderDataGenerator:
    """
    Generates voice data for reminder system training.
    
    Creates synthetic audio samples with characteristics matching
    different cognitive levels and dementia patterns.
    """
    
    def __init__(self, output_dir: str = "data/voice_samples"):
        """Initialize voice data generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize voice processor if available
        if VOICE_AVAILABLE:
            try:
                self.voice_processor = VoiceProcessor()
                logger.info("Voice processor initialized")
            except Exception as e:
                logger.warning(f"Voice processor failed to initialize: {e}")
                self.voice_processor = None
        else:
            self.voice_processor = None
        
        # Voice characteristics for different cognitive levels
        self.voice_characteristics = {
            "clear_confirmation": {
                "speech_rate": (120, 180),  # words per minute
                "pause_frequency": (0.1, 0.3),  # pauses per second
                "volume_variation": (0.1, 0.3),  # amplitude variation
                "pitch_variation": (0.8, 1.2),  # relative to baseline
                "articulation_clarity": (0.8, 1.0)
            },
            "mild_confusion": {
                "speech_rate": (90, 140),
                "pause_frequency": (0.3, 0.6),
                "volume_variation": (0.2, 0.5),
                "pitch_variation": (0.7, 1.3),
                "articulation_clarity": (0.6, 0.8)
            },
            "moderate_confusion": {
                "speech_rate": (70, 110),
                "pause_frequency": (0.5, 0.9),
                "volume_variation": (0.3, 0.7),
                "pitch_variation": (0.6, 1.4),
                "articulation_clarity": (0.4, 0.7)
            },
            "high_confusion": {
                "speech_rate": (50, 90),
                "pause_frequency": (0.7, 1.2),
                "volume_variation": (0.4, 0.8),
                "pitch_variation": (0.5, 1.5),
                "articulation_clarity": (0.2, 0.5)
            },
            "delay_resistance": {
                "speech_rate": (140, 200),
                "pause_frequency": (0.1, 0.4),
                "volume_variation": (0.2, 0.4),
                "pitch_variation": (0.9, 1.1),
                "articulation_clarity": (0.7, 0.9)
            }
        }
    
    def generate_voice_features(
        self, 
        text: str, 
        cognitive_level: str,
        audio_path: Optional[str] = None
    ) -> Dict:
        """
        Generate voice features for given text and cognitive level.
        
        If voice processing is available, generates actual audio.
        Otherwise, creates simulated features based on characteristics.
        """
        if self.voice_processor and audio_path:
            return self._generate_real_voice_features(text, cognitive_level, audio_path)
        else:
            return self._generate_simulated_voice_features(text, cognitive_level)
    
    def _generate_real_voice_features(
        self, 
        text: str, 
        cognitive_level: str,
        audio_path: str
    ) -> Dict:
        """Generate real voice features using TTS and processing."""
        try:
            # Generate synthetic audio using TTS (placeholder)
            # This would use a TTS engine to create audio from text
            audio_data = self._text_to_speech_synthetic(text, cognitive_level)
            
            # Save audio file
            audio_path = Path(audio_path)
            sf.write(audio_path, audio_data, 22050)  # 22kHz sample rate
            
            # Extract features using voice processor
            features = self.voice_processor.extract_features(str(audio_path))
            
            # Add cognitive-level adjustments
            features.update(self._adjust_features_for_cognitive_level(features, cognitive_level))
            
            return features
            
        except Exception as e:
            logger.warning(f"Real voice generation failed: {e}, falling back to simulation")
            return self._generate_simulated_voice_features(text, cognitive_level)
    
    def _generate_simulated_voice_features(self, text: str, cognitive_level: str) -> Dict:
        """Generate simulated voice features based on cognitive characteristics."""
        characteristics = self.voice_characteristics.get(cognitive_level, self.voice_characteristics["clear_confirmation"])
        
        # Text analysis for feature generation
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Generate features based on characteristics
        speech_rate = random.uniform(*characteristics["speech_rate"])
        pause_frequency = random.uniform(*characteristics["pause_frequency"])
        
        # Calculate derived features
        estimated_duration = (word_count / speech_rate) * 60  # seconds
        pause_count = int(pause_frequency * estimated_duration)
        
        # Voice quality features
        features = {
            # Temporal features
            "speech_rate_wpm": speech_rate,
            "total_duration_seconds": estimated_duration,
            "pause_count": pause_count,
            "pause_duration_ratio": pause_count * 0.5 / estimated_duration if estimated_duration > 0 else 0,
            "words_per_pause": word_count / (pause_count + 1),
            
            # Prosodic features
            "pitch_variance": random.uniform(*characteristics["pitch_variation"]),
            "volume_variance": random.uniform(*characteristics["volume_variation"]),
            "articulation_score": random.uniform(*characteristics["articulation_clarity"]),
            
            # Cognitive indicators
            "hesitation_ratio": min(pause_frequency / 0.5, 1.0),  # normalized hesitation
            "fluency_score": characteristics["articulation_clarity"][0],
            "speech_consistency": 1.0 - characteristics["volume_variation"][1],
            
            # Dementia-specific markers
            "semantic_fluency": self._calculate_semantic_fluency(text, cognitive_level),
            "word_finding_difficulty": self._calculate_word_finding_difficulty(text, cognitive_level),
            "repetition_score": self._calculate_repetition_score(text),
            
            # Audio quality (simulated)
            "signal_to_noise_ratio": random.uniform(15, 30),  # dB
            "spectral_centroid": random.uniform(1000, 3000),  # Hz
            "zero_crossing_rate": random.uniform(0.05, 0.15),
            
            # Metadata
            "cognitive_level": cognitive_level,
            "text_length": len(text),
            "word_count": word_count,
            "sentence_count": sentence_count
        }
        
        return features
    
    def _text_to_speech_synthetic(self, text: str, cognitive_level: str) -> np.ndarray:
        """
        Placeholder for TTS synthesis with cognitive characteristics.
        
        In a real implementation, this would:
        1. Use a TTS engine (e.g., gTTS, Azure Speech, Amazon Polly)
        2. Apply voice modifications based on cognitive level
        3. Add realistic speech patterns (hesitations, repetitions)
        """
        # Simulate audio generation
        characteristics = self.voice_characteristics[cognitive_level]
        
        # Estimate duration and generate placeholder audio
        word_count = len(text.split())
        speech_rate = random.uniform(*characteristics["speech_rate"])
        duration = (word_count / speech_rate) * 60  # seconds
        
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Generate synthetic audio (placeholder - replace with real TTS)
        # This creates a simple tone pattern as a placeholder
        t = np.linspace(0, duration, samples)
        frequency = random.uniform(80, 200)  # Base frequency
        audio = np.sin(2 * np.pi * frequency * t) * 0.1
        
        # Add characteristics based on cognitive level
        pitch_var = characteristics["pitch_variation"]
        volume_var = characteristics["volume_variation"]
        
        # Apply variations
        pitch_modulation = np.sin(2 * np.pi * 0.5 * t) * (pitch_var[1] - 1.0)
        audio *= (1.0 + pitch_modulation)
        
        volume_modulation = np.random.uniform(
            1.0 - volume_var[1], 
            1.0 + volume_var[1], 
            size=samples
        )
        audio *= volume_modulation
        
        # Add pauses based on pause frequency
        pause_freq = characteristics["pause_frequency"][1]
        pause_mask = np.random.random(samples) > pause_freq / 2
        audio *= pause_mask
        
        return audio.astype(np.float32)
    
    def _calculate_semantic_fluency(self, text: str, cognitive_level: str) -> float:
        """Calculate semantic fluency score based on text and cognitive level."""
        # Simple heuristic based on word diversity and coherence
        words = text.lower().split()
        unique_words = set(words)
        
        if len(words) == 0:
            return 0.0
        
        diversity_score = len(unique_words) / len(words)
        
        # Adjust based on cognitive level
        cognitive_multipliers = {
            "clear_confirmation": 1.0,
            "mild_confusion": 0.8,
            "moderate_confusion": 0.6,
            "high_confusion": 0.3,
            "delay_resistance": 0.9
        }
        
        multiplier = cognitive_multipliers.get(cognitive_level, 1.0)
        return diversity_score * multiplier
    
    def _calculate_word_finding_difficulty(self, text: str, cognitive_level: str) -> float:
        """Calculate word-finding difficulty score."""
        # Look for indicators of word-finding issues
        indicators = ["um", "uh", "what's", "you know", "thing", "stuff", "..."]
        words = text.lower().split()
        
        if len(words) == 0:
            return 0.0
        
        indicator_count = sum(1 for word in words if any(ind in word for ind in indicators))
        difficulty_score = indicator_count / len(words)
        
        # Adjust based on cognitive level
        cognitive_multipliers = {
            "clear_confirmation": 0.1,
            "mild_confusion": 0.3,
            "moderate_confusion": 0.6,
            "high_confusion": 0.8,
            "delay_resistance": 0.1
        }
        
        base_multiplier = cognitive_multipliers.get(cognitive_level, 0.3)
        return min(difficulty_score + base_multiplier, 1.0)
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score in the text."""
        words = text.lower().split()
        if len(words) <= 1:
            return 0.0
        
        # Count word repetitions
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        return repeated_words / len(words)
    
    def _adjust_features_for_cognitive_level(self, features: Dict, cognitive_level: str) -> Dict:
        """Adjust extracted features based on cognitive level."""
        adjustments = {}
        characteristics = self.voice_characteristics.get(cognitive_level, {})
        
        # Apply cognitive-level specific adjustments
        if "speech_rate_wpm" in features:
            rate_range = characteristics.get("speech_rate", (100, 150))
            # Nudge towards expected range
            current_rate = features["speech_rate_wpm"]
            target_rate = (rate_range[0] + rate_range[1]) / 2
            adjustments["speech_rate_wpm"] = (current_rate + target_rate) / 2
        
        return adjustments
    
    def generate_voice_dataset(
        self, 
        text_data_file: str,
        output_file: str = "data/voice_reminder_features.csv",
        audio_dir: Optional[str] = None
    ) -> str:
        """
        Generate voice features for existing text dataset.
        
        Args:
            text_data_file: CSV file with text reminder data
            output_file: Output file for voice features
            audio_dir: Optional directory to save audio files
            
        Returns:
            Path to generated voice features file
        """
        logger.info(f"Generating voice features from {text_data_file}")
        
        # Load text data
        df = pd.read_csv(text_data_file)
        
        if audio_dir:
            audio_path = Path(audio_dir)
            audio_path.mkdir(parents=True, exist_ok=True)
        
        voice_features_list = []
        
        for idx, row in df.iterrows():
            try:
                # Get text and cognitive level
                user_response = row.get('user_response', '')
                cognitive_level = row.get('cognitive_level', 'clear_confirmation')
                reminder_id = row.get('reminder_id', f'reminder_{idx}')
                
                # Generate audio path if audio directory provided
                audio_file_path = None
                if audio_dir:
                    audio_file_path = audio_path / f"{reminder_id}.wav"
                
                # Generate voice features
                voice_features = self.generate_voice_features(
                    user_response, 
                    cognitive_level,
                    str(audio_file_path) if audio_file_path else None
                )
                
                # Add metadata
                voice_features.update({
                    'reminder_id': reminder_id,
                    'original_text': user_response,
                    'audio_file_path': str(audio_file_path) if audio_file_path else None,
                    'generation_timestamp': pd.Timestamp.now().isoformat()
                })
                
                voice_features_list.append(voice_features)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(df)} samples")
                    
            except Exception as e:
                logger.warning(f"Failed to process row {idx}: {e}")
                continue
        
        # Save voice features
        voice_df = pd.DataFrame(voice_features_list)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        voice_df.to_csv(output_path, index=False)
        
        logger.info(f"Generated voice features for {len(voice_features_list)} samples")
        logger.info(f"Voice features saved to {output_path}")
        
        return str(output_path)
    
    def combine_text_and_voice_features(
        self,
        text_features_file: str,
        voice_features_file: str,
        output_file: str = "data/multimodal_reminder_dataset.csv"
    ) -> str:
        """
        Combine text and voice features into a single multimodal dataset.
        
        Args:
            text_features_file: CSV file with text features
            voice_features_file: CSV file with voice features
            output_file: Output file for combined dataset
            
        Returns:
            Path to combined dataset
        """
        logger.info("Combining text and voice features...")
        
        # Load datasets
        text_df = pd.read_csv(text_features_file)
        voice_df = pd.read_csv(voice_features_file)
        
        # Merge on reminder_id
        combined_df = pd.merge(
            text_df, 
            voice_df, 
            on='reminder_id', 
            how='inner',
            suffixes=('_text', '_voice')
        )
        
        # Add multimodal features
        combined_df = self._add_multimodal_features(combined_df)
        
        # Save combined dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Combined dataset saved to {output_path} with {len(combined_df)} samples")
        logger.info(f"Features: {len(combined_df.columns)} columns")
        
        return str(output_path)
    
    def _add_multimodal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that combine text and voice modalities."""
        
        # Cross-modal consistency features
        if 'cognitive_risk_score' in df.columns and 'fluency_score' in df.columns:
            df['text_voice_consistency'] = 1.0 - abs(
                df['cognitive_risk_score'] - (1.0 - df['fluency_score'])
            )
        
        # Combined confidence score
        if 'confidence' in df.columns and 'articulation_score' in df.columns:
            df['multimodal_confidence'] = (
                df['confidence'] * 0.6 + df['articulation_score'] * 0.4
            )
        
        # Response quality indicator
        text_quality_cols = ['semantic_incoherence', 'hesitation_pauses']
        voice_quality_cols = ['hesitation_ratio', 'word_finding_difficulty']
        
        if all(col in df.columns for col in text_quality_cols + voice_quality_cols):
            df['response_quality_score'] = (
                (1.0 - df['semantic_incoherence']) * 0.3 +
                (1.0 - df['hesitation_pauses'] / 10.0) * 0.2 +
                (1.0 - df['hesitation_ratio']) * 0.3 +
                (1.0 - df['word_finding_difficulty']) * 0.2
            )
        
        # Multimodal risk assessment
        risk_indicators = [
            'confusion_detected', 'memory_issue', 'hesitation_ratio', 'word_finding_difficulty'
        ]
        available_indicators = [col for col in risk_indicators if col in df.columns]
        
        if available_indicators:
            df['multimodal_risk_score'] = df[available_indicators].mean(axis=1)
        
        return df


def main():
    """Main function for voice data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate voice data for reminder system')
    parser.add_argument('--text-data', required=True, help='Input text data CSV file')
    parser.add_argument('--output-voice', default='data/voice_reminder_features.csv', help='Output voice features file')
    parser.add_argument('--output-combined', default='data/multimodal_reminder_dataset.csv', help='Output combined dataset file')
    parser.add_argument('--audio-dir', help='Directory to save generated audio files')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = VoiceReminderDataGenerator()
    
    # Generate voice features
    voice_features_file = generator.generate_voice_dataset(
        args.text_data,
        args.output_voice,
        args.audio_dir
    )
    
    # Combine with text features
    combined_file = generator.combine_text_and_voice_features(
        args.text_data,
        voice_features_file,
        args.output_combined
    )
    
    print(f"\n✅ Voice data generation complete!")
    print(f"Voice features: {voice_features_file}")
    print(f"Combined dataset: {combined_file}")


if __name__ == '__main__':
    main()