"""
Quick test script to verify the trained model works
"""
import sys
import pandas as pd
from src.models.conversational_ai.text_model_trainer import TextModelTrainer

def test_model():
    # Load the trained model
    print("Loading trained model...")
    trainer = TextModelTrainer()
    trainer.load_model('models/conversational_ai/trained_models/text_model.pkl')
    print("✓ Model loaded successfully!")
    
    # Load some test samples from the dataset
    print("\nLoading test data...")
    df = pd.read_csv('output/pitt_text_features.csv')
    print(f"✓ Loaded {len(df)} samples")
    
    # Get 5 random samples (Control and Dementia)
    control_samples = df[df['dementia_label'] == 0].sample(3, random_state=42)
    dementia_samples = df[df['dementia_label'] == 1].sample(3, random_state=42)
    
    print("\n" + "="*70)
    print("TESTING MODEL PREDICTIONS")
    print("="*70)
    
    print("\n📊 Control Group Samples (Should predict 0 - No Dementia):")
    print("-" * 70)
    for idx, row in control_samples.iterrows():
        features = row[['semantic_incoherence', 'repeated_questions', 'self_correction',
                       'low_confidence_answers', 'hesitation_pauses', 'emotion_slip',
                       'evening_errors']].values.reshape(1, -1)
        pred, prob = trainer.predict(features)
        dementia_prob = prob[0][1]
        print(f"  Sample {row['participant_id']:10s} → Prediction: {pred[0]} | Dementia Risk: {dementia_prob:.1%}")
    
    print("\n🧠 Dementia Group Samples (Should predict 1 - Dementia):")
    print("-" * 70)
    for idx, row in dementia_samples.iterrows():
        features = row[['semantic_incoherence', 'repeated_questions', 'self_correction',
                       'low_confidence_answers', 'hesitation_pauses', 'emotion_slip',
                       'evening_errors']].values.reshape(1, -1)
        pred, prob = trainer.predict(features)
        dementia_prob = prob[0][1]
        print(f"  Sample {row['participant_id']:10s} → Prediction: {pred[0]} | Dementia Risk: {dementia_prob:.1%}")
    
    print("\n" + "="*70)
    print("✓ Model test completed successfully!")
    print("="*70)

if __name__ == "__main__":
    test_model()
