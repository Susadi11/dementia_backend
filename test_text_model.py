#!/usr/bin/env python3
"""
Test Text Model Training and Inference

This script:
1. Trains text model using sample data
2. Tests inference with sample transcripts
3. Shows results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.conversational_ai import TextModelTrainer
from src.features.conversational_ai import TextProcessor

print("="*70)
print("TEXT MODEL TRAINING AND TESTING")
print("="*70)

# Step 1: Train model
print("\n[1] Training Text Model...")
print("-" * 70)

try:
    import pandas as pd
    import numpy as np

    # Load training data
    df = pd.read_csv('data/sample/text/text_features.csv')
    print(f"Loaded {len(df)} training samples")

    # Extract features and labels
    feature_cols = [
        'semantic_incoherence',
        'repeated_questions',
        'self_correction',
        'low_confidence_answers',
        'hesitation_pauses',
        'emotion_slip',
        'evening_errors'
    ]

    X = df[feature_cols].values
    y = df['dementia_label'].values

    print(f"Features shape: {X.shape}")
    print(f"Labels: {np.bincount(y)} (0=normal, 1=dementia)")

    # Train model
    trainer = TextModelTrainer(model_type='random_forest')
    metrics = trainer.train(X, y, test_size=0.3)

    print("\nTraining Complete!")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    # Save model
    trainer.save_model('models/text_model_sample.pkl')
    print("\nModel saved to: models/text_model_sample.pkl")

except Exception as e:
    print(f"Error training: {e}")
    sys.exit(1)

# Step 2: Test inference on sample transcripts
print("\n" + "="*70)
print("[2] Testing Inference on Sample Transcripts")
print("="*70)

processor = TextProcessor()

# Load trained model
trainer_loaded = TextModelTrainer()
trainer_loaded.load_model('models/text_model_sample.pkl')

# Test 1: Dementia transcript
print("\n[Test 1] Dementia Patient Transcript")
print("-" * 70)
try:
    with open('data/sample/text/sample_dementia_transcript.txt', 'r') as f:
        dementia_text = f.read()

    print("Extracting features from transcript...")
    features = processor.process(dementia_text)

    print("\nExtracted Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.4f}")

    prediction, confidence = trainer_loaded.predict_from_dict(features)

    print(f"\nPrediction: {'DEMENTIA RISK' if prediction == 1 else 'NO DEMENTIA RISK'}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

except Exception as e:
    print(f"Error: {e}")

# Test 2: Normal transcript
print("\n[Test 2] Normal Patient Transcript")
print("-" * 70)
try:
    with open('data/sample/text/sample_normal_transcript.txt', 'r') as f:
        normal_text = f.read()

    print("Extracting features from transcript...")
    features = processor.process(normal_text)

    print("\nExtracted Features:")
    for feature, value in features.items():
        print(f"  {feature}: {value:.4f}")

    prediction, confidence = trainer_loaded.predict_from_dict(features)

    print(f"\nPrediction: {'DEMENTIA RISK' if prediction == 1 else 'NO DEMENTIA RISK'}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print("\nNext steps:")
print("1. Train with your own data:")
print("   python scripts/train_text_model.py --data-file data/your_data.csv --model-output models/your_model.pkl")
print("\n2. Run inference on new transcripts:")
print("   python scripts/inference_text.py --model models/your_model.pkl --text-file data/new_transcript.txt")
print("="*70)
