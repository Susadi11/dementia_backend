"""
Test the Enhanced Reminder System with integrated Pitt Corpus models.

This script tests the updated reminder system to ensure it's using
the enhanced models trained with both synthetic and real dementia data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.features.reminder_system.reminder_analyzer import PittBasedReminderAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_system():
    """Test the enhanced reminder system with sample responses."""
    
    print("🧪 TESTING ENHANCED REMINDER SYSTEM")
    print("=" * 60)
    
    try:
        # Initialize analyzer (should load enhanced models by default)
        analyzer = PittBasedReminderAnalyzer()
        
        # Test responses that should trigger different model outputs
        test_responses = [
            {
                'text': "I already took my medicine this morning",
                'description': "Clear confirmation - should show low risk"
            },
            {
                'text': "Um... what medicine? I don't remember taking any pills",
                'description': "Memory issue - should show high risk and confusion"
            },
            {
                'text': "I think I did... maybe? I'm not sure",
                'description': "Uncertainty - should show moderate risk"
            },
            {
                'text': "What? What are you talking about?",
                'description': "Confusion - should trigger caregiver alert"
            }
        ]
        
        print(f"📊 Testing {len(test_responses)} sample responses...\n")
        
        for i, test_case in enumerate(test_responses, 1):
            print(f"🔍 TEST {i}: {test_case['description']}")
            print(f"📝 Response: \"{test_case['text']}\"")
            
            # Analyze response
            result = analyzer.analyze_reminder_response(test_case['text'])
            
            # Display results
            print(f"🧠 Cognitive Risk: {result['cognitive_risk_score']:.3f}")
            print(f"❓ Confusion Detected: {result['confusion_detected']}")
            print(f"💭 Memory Issue: {result['memory_issue_detected']}")
            print(f"⚠️  Caregiver Alert: {result['caregiver_alert_needed']}")
            print(f"🎯 Model Type: {result['model_type']}")
            print(f"📈 Confidence: {result['confidence']:.3f}")
            
            # Show enhanced predictions if available
            if result.get('enhanced_predictions'):
                print("🤖 Enhanced Model Predictions:")
                for model_name, pred_data in result['enhanced_predictions'].items():
                    print(f"   - {model_name}: {pred_data['prediction']} (conf: {pred_data['confidence']:.3f})")
            
            print("-" * 50)
        
        # Summary
        if analyzer.enhanced_models:
            model_info = analyzer.enhanced_models.get_model_info()
            print("\n✅ ENHANCED SYSTEM STATUS:")
            print(f"🏆 Models Loaded: {', '.join(model_info['models_loaded'])}")
            print(f"📊 Training Samples: {model_info['total_samples']}")
            print(f"📅 Training Date: {model_info['training_date']}")
            print("\n🎉 Enhanced system is working correctly!")
        else:
            print("\n⚠️  WARNING: Enhanced models not loaded, using fallback system")
            
    except Exception as e:
        print(f"❌ Error testing enhanced system: {e}")
        logger.error(f"Error testing enhanced system: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    success = test_enhanced_system()
    if success:
        print("\n🚀 NEXT STEP: Your enhanced models are ready for production!")
        print("   Update your API server to use the enhanced system:")
        print("   python run_api.py")
    else:
        print("\n❌ Enhanced system test failed. Check logs for details.")