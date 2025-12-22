"""
Integration Status Summary

Shows the current status of your Pitt Corpus integration.
"""

import pandas as pd
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Show integration status."""
    print("="*60)
    print("PITT CORPUS INTEGRATION STATUS")
    print("="*60)
    
    # Check data integration
    enhanced_data_file = Path("data/enhanced_training_data.csv")
    if enhanced_data_file.exists():
        df = pd.read_csv(enhanced_data_file)
        print(f"✅ Enhanced dataset created: {len(df):,} samples")
        
        if 'data_source' in df.columns:
            source_counts = df['data_source'].value_counts()
            print(f"   - Synthetic samples: {source_counts.get('synthetic', 0):,}")
            print(f"   - Pitt Corpus samples: {source_counts.get('pitt_corpus', 0):,}")
        
        print(f"   - Features: {len(df.columns)} columns")
        
        # Show sample of Pitt data
        pitt_samples = df[df['data_source'] == 'pitt_corpus'].head(3)
        if not pitt_samples.empty:
            print("\nSample Pitt Corpus data:")
            for idx, row in pitt_samples.iterrows():
                task = row.get('task_type', 'unknown')
                participant = row.get('participant_id', 'unknown')
                print(f"   - Participant {participant}, Task: {task}")
    else:
        print("❌ Enhanced dataset not found")
    
    # Check model training
    models_dir = Path("models/reminder_system")
    enhanced_metadata = models_dir / "enhanced_training_metadata.json"
    
    if enhanced_metadata.exists():
        print("\n✅ Enhanced models trained:")
        
        with open(enhanced_metadata, 'r') as f:
            metadata = json.load(f)
        
        training_results = metadata.get('training_results', {})
        for model_name, result in training_results.items():
            if 'error' in result:
                print(f"   ❌ {model_name}: {result['error']}")
            else:
                score = result.get('best_score', 0)
                print(f"   ✅ {model_name}: {score:.3f}")
        
        print(f"\nTraining date: {metadata.get('training_date', 'unknown')}")
        print(f"Total samples used: {metadata.get('total_samples', 0):,}")
    else:
        print("\n❌ Enhanced models not trained")
    
    # Check model files
    model_files = [
        "confusion_detection_model.joblib",
        "cognitive_risk_model.joblib", 
        "caregiver_alert_model.joblib",
        "response_classifier_model.joblib"
    ]
    
    print("\nModel files:")
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            size_kb = model_path.stat().st_size / 1024
            print(f"   ✅ {model_file} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {model_file} missing")
    
    # Integration quality assessment
    print("\n" + "="*60)
    print("INTEGRATION QUALITY ASSESSMENT")
    print("="*60)
    
    if enhanced_data_file.exists():
        df = pd.read_csv(enhanced_data_file)
        
        # Data quality metrics
        pitt_samples = len(df[df['data_source'] == 'pitt_corpus'])
        synthetic_samples = len(df[df['data_source'] == 'synthetic'])
        
        print(f"Data Balance:")
        print(f"   - Pitt Corpus: {pitt_samples:,} ({pitt_samples/len(df)*100:.1f}%)")
        print(f"   - Synthetic: {synthetic_samples:,} ({synthetic_samples/len(df)*100:.1f}%)")
        
        # Feature completeness
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.1]
        
        if len(high_missing) == 0:
            print("✅ Data quality: Good (low missing values)")
        else:
            print(f"⚠️  Data quality: {len(high_missing)} features with >10% missing")
        
        # Dementia representation
        if 'dementia_label' in df.columns:
            dementia_counts = df['dementia_label'].value_counts()
            print(f"Dementia representation:")
            print(f"   - Control (0): {dementia_counts.get(0, 0):,}")
            print(f"   - Dementia (1): {dementia_counts.get(1, 0):,}")
    
    # Next steps
    print("\n" + "="*60)
    print("RECOMMENDED NEXT STEPS")
    print("="*60)
    
    if enhanced_data_file.exists() and enhanced_metadata.exists():
        print("🎉 Integration successful! Your models now include real-world data.")
        print("\nNext steps:")
        print("1. Deploy enhanced models in your reminder system")
        print("2. Monitor real-world performance improvements")
        print("3. Compare with previous synthetic-only models")
        print("4. Collect user feedback on system accuracy")
        
        print("\nKey improvements you should see:")
        print("- Better detection of real confusion patterns")
        print("- More accurate cognitive risk assessment")
        print("- Fewer false positives/negatives")
        print("- Better adaptation to individual speech patterns")
    else:
        print("⚠️  Integration incomplete. Run:")
        print("python scripts/run_integration_fixed.py")


if __name__ == "__main__":
    main()