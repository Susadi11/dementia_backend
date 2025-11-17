#!/usr/bin/env python3
"""
Inference Script for Voice-Based Dementia Detection

Usage:
    python scripts/inference_voice.py --model <path/to/voice_model.pkl> --audio <path/to/audio.wav>

Or with feature dictionary:
    python scripts/inference_voice.py --model <path/to/voice_model.pkl> --features <path/to/features.json>
"""

import argparse
import logging
import json
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.conversational_ai import VoiceModelTrainer, VOICE_FEATURES
from src.features.conversational_ai import VoiceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_voice_inference(model_path: str, audio_path: str = None, features_dict: dict = None):
    """
    Run inference using trained voice model.

    Args:
        model_path: Path to trained voice model
        audio_path: Path to audio file (if features not provided)
        features_dict: Dictionary with pre-computed voice features

    Returns:
        Tuple of (prediction, confidence)
    """
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        trainer = VoiceModelTrainer()
        trainer.load_model(model_path)
        logger.info("Model loaded successfully")

        # Get features
        if features_dict is None:
            if audio_path is None:
                raise ValueError("Either audio_path or features_dict must be provided")

            logger.info(f"Analyzing audio: {audio_path}")
            analyzer = VoiceAnalyzer()
            features_dict = analyzer.analyze(audio_path=audio_path)
            logger.info(f"Extracted voice features: {features_dict}")
        else:
            logger.info(f"Using provided features: {features_dict}")

        # Make prediction
        prediction, confidence = trainer.predict_from_dict(features_dict)

        result = {
            'prediction': prediction,
            'prediction_label': 'Dementia Risk' if prediction == 1 else 'No Dementia Risk',
            'confidence': confidence,
            'features': features_dict,
            'feature_names': VOICE_FEATURES
        }

        logger.info(f"Prediction: {result['prediction_label']}")
        logger.info(f"Confidence: {confidence:.4f}")

        return result

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with voice-based dementia detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference from audio file
  python scripts/inference_voice.py --model models/voice_model.pkl --audio data/sample_audio.wav

  # Inference from precomputed features JSON
  python scripts/inference_voice.py --model models/voice_model.pkl --features data/voice_features.json

Features expected:
  - vocal_tremors (0-1)
  - slowed_speech (0-1)
  - in_session_decline (0-1)
        """
    )

    parser.add_argument('--model', required=True, help='Path to trained voice model')
    parser.add_argument('--audio', help='Path to audio file for analysis')
    parser.add_argument('--features', help='Path to JSON file with precomputed features')

    args = parser.parse_args()

    features_dict = None
    if args.features:
        try:
            with open(args.features, 'r') as f:
                features_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            sys.exit(1)

    result = run_voice_inference(
        model_path=args.model,
        audio_path=args.audio,
        features_dict=features_dict
    )

    if result:
        print("\n" + "="*60)
        print("VOICE MODEL INFERENCE RESULT")
        print("="*60)
        print(f"Prediction:  {result['prediction_label']}")
        print(f"Confidence:  {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print("\nFeatures extracted:")
        for feature, value in result['features'].items():
            print(f"  {feature}: {value:.4f}")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
