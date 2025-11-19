#!/usr/bin/env python3
"""
Training Script for Voice-Based Dementia Detection Model

Usage:
    python scripts/train_voice_model.py --data-file <path/to/data.csv> --model-output <path/to/model.pkl> --model-type random_forest

Features used:
    - vocal_tremors: Amplitude modulation intensity (0-1)
    - slowed_speech: Speech rate reduction indicator (0-1)
    - in_session_decline: Progressive fatigue during session (0-1)
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.conversational_ai import VoiceModelTrainer, VOICE_FEATURES
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_voice_model(data_file: str, model_output: str, model_type: str = 'random_forest', test_size: float = 0.2, kfold: int = 0):
    """
    Train voice model for dementia detection.

    Args:
        data_file: Path to CSV file with training data
        model_output: Path to save trained model
        model_type: Type of model ('random_forest', 'gradient_boost', 'logistic')
        test_size: Test/train split ratio
    """
    logger.info(f"Starting voice model training from {data_file}")

    try:
        # Load data
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} samples from {data_file}")

        # Extract voice features
        X_data = []
        for idx, row in df.iterrows():
            feature_row = [row.get(feat, 0.0) for feat in VOICE_FEATURES]
            X_data.append(feature_row)

        X = np.array(X_data)
        y = np.array(df['dementia_label'].values if 'dementia_label' in df.columns else df['label'].values)

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Voice features: {VOICE_FEATURES}")

        trainer = VoiceModelTrainer(model_type=model_type)

        if kfold and int(kfold) > 1:
            logger.info(f"Running {kfold}-fold cross-validation for voice model")
            skf = StratifiedKFold(n_splits=int(kfold), shuffle=True, random_state=42)
            accs = []; precs = []; recs = []; f1s = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                trainer.train(X_train, y_train, test_size=0.0)
                preds, probs = trainer.predict(X_test)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                logger.info(f"Fold {fold} - acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
                accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

            import numpy as _np
            logger.info(f"CV mean accuracy: {_np.mean(accs):.4f} ± {_np.std(accs):.4f}")
            logger.info(f"CV mean precision: {_np.mean(precs):.4f} ± {_np.std(precs):.4f}")
            logger.info(f"CV mean recall: {_np.mean(recs):.4f} ± {_np.std(recs):.4f}")
            logger.info(f"CV mean f1: {_np.mean(f1s):.4f} ± {_np.std(f1s):.4f}")

            # Retrain on full data and save
            trainer.train(X, y, test_size=0.0)
            trainer.save_model(model_output)
            logger.info(f"Final voice model trained on full data and saved to {model_output}")
            return True

        else:
            metrics = trainer.train(X, y, test_size=test_size)

            logger.info(f"Training complete!")
            logger.info(f"Model Metrics:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1']:.4f}")

            trainer.save_model(model_output)
            logger.info(f"Model saved to {model_output}")
            return True

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train voice-based dementia detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Train with default settings
  python scripts/train_voice_model.py --data-file data/voice_features.csv --model-output models/voice_model.pkl

  # Train with gradient boost
  python scripts/train_voice_model.py --data-file data/voice_features.csv --model-output models/voice_model.pkl --model-type gradient_boost

  # Custom test split
  python scripts/train_voice_model.py --data-file data/voice_features.csv --model-output models/voice_model.pkl --test-size 0.3

Voice features used: {', '.join(VOICE_FEATURES)}
        """
    )

    parser.add_argument('--data-file', required=True, help='Path to training data CSV file')
    parser.add_argument('--model-output', required=True, help='Path to save trained model')
    parser.add_argument('--model-type', default='random_forest',
                       choices=['random_forest', 'gradient_boost', 'logistic'],
                       help='Type of model to train (default: random_forest)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--kfold', type=int, default=0,
                       help='If >1, run k-fold CV and retrain on full data (default: 0 - disabled)')

    args = parser.parse_args()

    success = train_voice_model(
        data_file=args.data_file,
        model_output=args.model_output,
        model_type=args.model_type,
        test_size=args.test_size,
        kfold=args.kfold
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
