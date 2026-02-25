"""
Test Dementia Feature Extraction

This script demonstrates how to access and analyze dementia detection features
from user responses. Run this to see the features in action.

Usage:
    python test_dementia_features.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.features.conversational_ai.feature_extractor import FeatureExtractor
from src.features.reminder_system.reminder_analyzer import PittBasedReminderAnalyzer
from src.utils.helpers import calculate_overall_risk


def print_separator(char="=", length=70):
    """Print a separator line."""
    print(char * length)


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_feature_extraction():
    """Test extracting dementia features from different response types."""
    
    print_section("🧪 DEMENTIA FEATURE EXTRACTION TEST")
    
    # Initialize feature extractor
    print("📦 Initializing Feature Extractor...")
    extractor = FeatureExtractor(use_nlp=True)
    print("✅ Feature Extractor ready\n")
    
    # Test cases representing different cognitive states
    test_cases = [
        {
            "name": "Healthy Response",
            "text": "Yes, I took my blood pressure medication this morning at 8 AM.",
            "expected": "Low cognitive risk - clear, confident response"
        },
        {
            "name": "Moderate Confusion",
            "text": "Um... I think I took it? Maybe the blue one? I'm not really sure.",
            "expected": "Medium risk - uncertainty and filler words"
        },
        {
            "name": "High Confusion",
            "text": "What medicine? I don't... um... what was I supposed to take?",
            "expected": "High risk - repeated questions, memory issues"
        },
        {
            "name": "Memory Impairment",
            "text": "I forgot. I don't remember what you're talking about.",
            "expected": "High risk - explicit memory markers"
        },
        {
            "name": "Repetitive Speech",
            "text": "The pill, the pill... I think the pill was... was it the blue pill?",
            "expected": "Medium-High risk - repetitions and word-finding difficulty"
        },
        {
            "name": "Incoherent Response",
            "text": "The... yesterday... blue... I think... maybe later... not sure.",
            "expected": "High risk - semantic incoherence"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'─'*70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'─'*70}")
        print(f"📝 Response: \"{test_case['text']}\"")
        print(f"Expected: {test_case['expected']}\n")
        
        # Extract features
        features = extractor.extract_features(transcript_text=test_case['text'])
        
        # Calculate overall cognitive risk
        cognitive_risk = calculate_overall_risk(features)
        
        # Display results
        print("📊 EXTRACTED DEMENTIA FEATURES:")
        print(f"   • Filler Words Ratio:     {features.get('filler_words', 0):.3f}")
        print(f"   • Repetitions:            {features.get('repetitions', 0)}")
        print(f"   • Semantic Incoherence:   {features.get('semantic_incoherence', 0):.3f}")
        print(f"   • Memory Markers:         {features.get('memory_markers', 0)}")
        print(f"   • Confusion Markers:      {features.get('confusion_markers', 0)}")
        print(f"   • Word Count:             {features.get('word_count', 0)}")
        print(f"   • Unique Words:           {features.get('unique_words', 0)}")
        print(f"   • Lexical Diversity:      {features.get('lexical_diversity', 0):.3f}")
        print(f"   • Hesitation Pauses:      {features.get('hesitation_pauses', 0):.3f}")
        
        # Risk assessment
        print(f"\n🎯 COGNITIVE RISK SCORE: {cognitive_risk:.3f}")
        
        if cognitive_risk < 0.3:
            risk_level = "🟢 LOW RISK - Normal cognitive function"
        elif cognitive_risk < 0.6:
            risk_level = "🟡 MEDIUM RISK - Some cognitive concerns"
        elif cognitive_risk < 0.7:
            risk_level = "🟠 ELEVATED RISK - Monitor closely"
        else:
            risk_level = "🔴 HIGH RISK - Caregiver alert recommended"
        
        print(f"   Assessment: {risk_level}\n")


def test_reminder_analyzer():
    """Test the full reminder analyzer with enhanced models."""
    
    print_section("🤖 PITT-BASED REMINDER ANALYZER TEST")
    
    print("📦 Initializing Pitt-Based Analyzer...")
    print("   Loading Enhanced Models trained on DementiaBank Pitt Corpus...")
    analyzer = PittBasedReminderAnalyzer(use_enhanced_models=True)
    print("✅ Analyzer ready\n")
    
    # Test response
    user_response = "Um... I don't remember. What medicine was it again?"
    
    print(f"📝 Analyzing Response:")
    print(f"   \"{user_response}\"\n")
    
    # Analyze with full context
    result = analyzer.analyze_reminder_response(
        user_response=user_response,
        reminder_context={
            'priority': 'critical',
            'category': 'medication',
            'title': 'Take morning blood pressure medication'
        }
    )
    
    print("📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print(f"   • Cognitive Risk Score:    {result['cognitive_risk_score']:.3f}")
    print(f"   • Confusion Detected:      {result['confusion_detected']}")
    print(f"   • Memory Issue Detected:   {result['memory_issue_detected']}")
    print(f"   • Uncertainty Detected:    {result['uncertainty_detected']}")
    print(f"   • Interaction Type:        {result['interaction_type']}")
    print(f"   • Model Confidence:        {result['confidence']:.3f}")
    print(f"   • Model Type:              {result['model_type']}")
    
    print(f"\n🎯 RECOMMENDED ACTION:")
    print(f"   {result['recommended_action']}")
    
    print(f"\n🚨 CAREGIVER ALERT:")
    if result['caregiver_alert_needed']:
        print(f"   ⚠️  YES - Caregiver should be notified immediately")
    else:
        print(f"   ✅ NO - No immediate alert needed")
    
    # Show enhanced model predictions if available
    if result.get('enhanced_predictions'):
        print(f"\n🧠 ENHANCED MODEL PREDICTIONS:")
        for model_name, prediction in result['enhanced_predictions'].items():
            print(f"   • {model_name}:")
            print(f"     - Prediction: {prediction['prediction']}")
            print(f"     - Confidence: {prediction['confidence']:.3f}")


def test_api_simulation():
    """Simulate what the API endpoint returns."""
    
    print_section("🌐 API ENDPOINT SIMULATION")
    
    print("This simulates what you would get from:")
    print("POST /api/reminders/respond\n")
    
    analyzer = PittBasedReminderAnalyzer(use_enhanced_models=True)
    
    response_text = "I think I... um... I forgot which pill"
    
    print(f"Request Body:")
    print(f"{{")
    print(f'  "reminder_id": "rem_12345",')
    print(f'  "user_id": "patient_001",')
    print(f'  "response_text": "{response_text}"')
    print(f"}}\n")
    
    result = analyzer.analyze_reminder_response(
        user_response=response_text,
        reminder_context={'priority': 'critical', 'category': 'medication'}
    )
    
    print(f"Response (JSON):")
    print(f"{{")
    print(f'  "status": "success",')
    print(f'  "analysis": {{')
    print(f'    "cognitive_risk_score": {result["cognitive_risk_score"]:.2f},')
    print(f'    "confusion_detected": {str(result["confusion_detected"]).lower()},')
    print(f'    "memory_issue_detected": {str(result["memory_issue_detected"]).lower()},')
    print(f'    "interaction_type": "{result["interaction_type"]}",')
    print(f'    "recommended_action": "{result["recommended_action"]}",')
    print(f'    "caregiver_alert_needed": {str(result["caregiver_alert_needed"]).lower()},')
    print(f'    "features": {{')
    
    features = result['features']
    feature_items = list(features.items())[:5]  # Show first 5 features
    for key, value in feature_items:
        print(f'      "{key}": {value:.3f},')
    print(f'      ...')
    print(f'    }}')
    print(f'  }}')
    print(f"}}\n")


def main():
    """Run all tests."""
    
    print("\n" + "="*70)
    print("  DEMENTIA DETECTION FEATURES - COMPREHENSIVE TEST")
    print("  Context-Aware Smart Reminder System")
    print("="*70)
    
    try:
        # Test 1: Basic feature extraction
        test_feature_extraction()
        
        # Test 2: Full analyzer with models
        test_reminder_analyzer()
        
        # Test 3: API simulation
        test_api_simulation()
        
        print_section("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("\n💡 Next Steps:")
        print("   1. Start the API: python run_api.py")
        print("   2. Test endpoints: See HOW_TO_GET_DEMENTIA_FEATURES.md")
        print("   3. Integrate with mobile app")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
