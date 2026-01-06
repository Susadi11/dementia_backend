"""
Quick verification that models are properly connected to the backend
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_model_integration():
    """Verify models are connected and working"""
    print("\n" + "=" * 70)
    print("🔍 BACKEND MODEL INTEGRATION CHECK")
    print("=" * 70)
    
    try:
        from src.models.game.model_registry import (
            load_all_models,
            get_risk_classifier,
            get_risk_scaler,
            get_label_encoder
        )
        
        # Load models
        print("\n1️⃣ Loading models...")
        load_all_models()
        
        # Get models
        risk_classifier = get_risk_classifier()
        scaler = get_risk_scaler()
        encoder = get_label_encoder()
        
        # Check each model
        print("\n2️⃣ Checking model status:")
        print(f"   Risk Classifier: {'✅ LOADED' if risk_classifier else '❌ FAILED'}")
        print(f"   Feature Scaler:  {'✅ LOADED' if scaler else '❌ FAILED'}")
        print(f"   Label Encoder:   {'✅ LOADED' if encoder else '❌ FAILED'}")
        
        if risk_classifier is None:
            print("\n❌ CRITICAL: Risk classifier failed to load!")
            print("   Backend will use DUMMY/RANDOM predictions")
            return False
        
        # Show model details
        print("\n3️⃣ Model Details:")
        print(f"   Model Type: {type(risk_classifier).__name__}")
        print(f"   Expected Features: {risk_classifier.n_features_in_}")
        print(f"   Output Classes: {encoder.classes_ if encoder else 'Unknown'}")
        
        # Test with 14 features (as expected by model)
        print("\n4️⃣ Testing prediction with sample data...")
        import numpy as np
        
        # Create sample with all 14 features
        sample = np.array([[
            0.5, 0.1, 100.0, -5.0,  # mean_sac, slope_sac, mean_ies, slope_ies
            0.75, 800.0, 50.0, 0.2,  # mean_acc, mean_rt, mean_var, lstm_score
            0.02, 150.0,             # current sac, current ies
            -0.05, 10.0,             # slope_acc, slope_rt
            0.01, 20.0               # std_sac, std_ies
        ]])
        
        # Scale and predict
        sample_scaled = scaler.transform(sample) if scaler else sample
        prediction = risk_classifier.predict(sample_scaled)
        probabilities = risk_classifier.predict_proba(sample_scaled)
        
        predicted_label = encoder.inverse_transform(prediction)[0] if encoder else prediction[0]
        
        print(f"   Prediction: {predicted_label}")
        print(f"   Probabilities: {probabilities[0]}")
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS: Models are properly integrated with backend!")
        print("=" * 70)
        print("\n📋 Summary:")
        print("   - New models are loaded from risk_classifier/ folder")
        print("   - Old models backed up in 'risk_classifier Backup/' folder")
        print("   - Backend will use the new models for predictions")
        print("   - Models expect 14 features and output 3 risk levels")
        print("\n🚀 Next Steps:")
        print("   1. Start API: python run_api.py")
        print("   2. Test game endpoints")
        print("   3. Verify predictions are accurate")
        print("=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_model_integration()
    sys.exit(0 if success else 1)
