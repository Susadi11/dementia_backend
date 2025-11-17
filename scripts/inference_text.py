#!/usr/bin/env python3
"""
Inference Script for Text-Based Dementia Detection

Usage:
    python scripts/inference_text.py --model <path/to/text_model.pkl> --text <"transcript text">

Or with file:
    python scripts/inference_text.py --model <path/to/text_model.pkl> --text-file <path/to/transcript.txt>

Or with feature dictionary:
    python scripts/inference_text.py --model <path/to/text_model.pkl> --features <path/to/features.json>
"""

import argparse
import logging
import json
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.conversational_ai import TextModelTrainer, TEXT_FEATURES
from src.features.conversational_ai import TextProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_text_inference(model_path: str, text: str = None, text_file: str = None, features_dict: dict = None):
    """
    Run inference using trained text model.

    Args:
        model_path: Path to trained text model
        text: Text transcript as string
        text_file: Path to text file with transcript
        features_dict: Dictionary with pre-computed text features

    Returns:
        Dictionary with prediction and confidence
    """
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        trainer = TextModelTrainer()
        trainer.load_model(model_path)
        logger.info("Model loaded successfully")

        # Get features
        if features_dict is None:
            # Get transcript text
            if text_file:
                logger.info(f"Reading transcript from {text_file}")
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif text is None:
                raise ValueError("One of text, text_file, or features_dict must be provided")

            logger.info(f"Processing transcript ({len(text)} characters)")
            processor = TextProcessor()
            features_dict = processor.process(text)
            logger.info(f"Extracted text features: {features_dict}")
        else:
            logger.info(f"Using provided features: {features_dict}")

        # Make prediction
        prediction, confidence = trainer.predict_from_dict(features_dict)

        result = {
            'prediction': prediction,
            'prediction_label': 'Dementia Risk' if prediction == 1 else 'No Dementia Risk',
            'confidence': confidence,
            'features': features_dict,
            'feature_names': TEXT_FEATURES
        }

        logger.info(f"Prediction: {result['prediction_label']}")
        logger.info(f"Confidence: {confidence:.4f}")

        return result

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with text-based dementia detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference from text string
  python scripts/inference_text.py --model models/text_model.pkl --text "Your transcript text here..."

  # Inference from text file
  python scripts/inference_text.py --model models/text_model.pkl --text-file data/transcript.txt

  # Inference from precomputed features JSON
  python scripts/inference_text.py --model models/text_model.pkl --features data/text_features.json

Features extracted:
  - semantic_incoherence (0-1)
  - repeated_questions (count)
  - self_correction (count)
  - low_confidence_answers (0-1)
  - hesitation_pauses (count)
  - emotion_slip (0-1)
  - evening_errors (0-1)
        """
    )

    parser.add_argument('--model', required=True, help='Path to trained text model')
    parser.add_argument('--text', help='Text transcript to analyze')
    parser.add_argument('--text-file', help='Path to text file with transcript')
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

    result = run_text_inference(
        model_path=args.model,
        text=args.text,
        text_file=args.text_file,
        features_dict=features_dict
    )

    if result:
        print("\n" + "="*60)
        print("TEXT MODEL INFERENCE RESULT")
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
