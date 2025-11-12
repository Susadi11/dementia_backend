"""
Test Script - Dementia Detection Using Sample Data

This script demonstrates how to:
1. Load sample data
2. Extract features from text and audio
3. Make dementia risk predictions
4. Generate reports

Usage:
    python test_prediction.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.conversational_ai.feature_extractor import FeatureExtractor
from preprocessing.data_loader import SampleDataLoader, DatasetManager


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("DEMENTIA DETECTION SYSTEM - TEST WITH SAMPLE DATA")
    print("="*80)

    print("\nLoading Sample Dataset...")
    dataset_manager = DatasetManager()
    dataset_manager.print_dataset_info()

    print("\nInitializing Feature Extractor...")
    feature_extractor = FeatureExtractor()

    component_info = feature_extractor.get_component_info()
    print("\nAvailable Components:")
    print(f"   - Text Processor: {component_info['text_processor']['description']}")
    print(f"   - Voice Analyzer: {component_info['voice_analyzer']['description']}")
    print(f"   - Total Features: {len(component_info['all_features'])}")

    print("\n" + "="*80)
    print("ANALYZING SAMPLE CASES")
    print("="*80)

    samples = dataset_manager.get_all_samples()

    for sample in samples:
        sample_id = sample.get('id')
        label = sample.get('label')
        age = sample.get('age')
        gender = sample.get('gender')

        print(f"\n{'─'*80}")
        print(f"Sample: {sample_id}")
        print(f"   Patient Info: Age {age}, Gender {gender}")
        print(f"   True Label: {label.upper()}")
        print(f"{'─'*80}")

        try:
            transcript_path = str(sample.get('transcript_path'))
            audio_path = str(sample.get('audio_path'))

            print(f"\n   Processing Transcript: {Path(transcript_path).name}")
            print(f"   Processing Audio: {Path(audio_path).name if Path(audio_path).exists() else '(not found)'}")

            features = feature_extractor.extract_features_normalized(
                transcript_path=transcript_path,
                audio_path=audio_path if Path(audio_path).exists() else None
            )

            report = feature_extractor.get_feature_report(features)
            print(report)

            expected_features = sample.get('features', {})
            if expected_features:
                print("   Validation against expected values:")
                for feat_name, expected_value in expected_features.items():
                    extracted_value = features.get(feat_name, 0.0)
                    match = "[OK]" if abs(extracted_value - expected_value) < 0.2 else "[WARN]"
                    print(f"      {match} {feat_name}: extracted={extracted_value:.3f}, expected={expected_value:.3f}")

        except Exception as e:
            print(f"   Error processing sample: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)

    stats = dataset_manager.sample_loader.get_sample_statistics()
    print(f"\nTotal Cases: {stats.get('total_samples')}")
    print(f"  - Control: {stats.get('control_count')}")
    print(f"  - Dementia Risk: {stats.get('dementia_risk_count')}")
    print(f"\nAge Statistics:")
    print(f"  - Mean: {stats.get('age_mean', 0):.1f} years")
    print(f"  - Range: {stats.get('age_range', (0, 0))}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. To use real data when available:
   - Place your dataset in ./data/real/
   - Create metadata file: ./data/real/metadata/dataset.json
   - Update DatasetManager to point to real data
   - All code will work seamlessly with real data

2. To start the API server:
   - Run: python run_api.py
   - Access API docs at: http://localhost:8000/docs

3. To train a prediction model:
   - Extract features from all samples
   - Train with: src/models/dementia_predictor.py
   - Model will use rule-based prediction with sample data
   - Switch to ML models when real dataset is available

4. Features Captured:
   ✓ Text-based (7 parameters):
     - Semantic incoherence, Repeated questions, Self-correction
     - Low-confidence answers, Hesitation pauses, Emotion+slip
     - Evening errors

   ✓ Voice-based (3 parameters):
     - Vocal tremors, Slowed speech, In-session decline

No changes needed when switching from sample to real data!
    """)

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
