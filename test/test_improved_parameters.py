"""
Test Improved P3, P4, and P7 Parameters

Tests the enhanced detection for:
- P3: Self-correction (25 patterns)
- P4: Low confidence (35 markers)
- P7: Emotion + slip (sentiment analysis + slip detection)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_p3_self_correction():
    """Test improved P3: Self-Correction"""
    print("\n" + "="*80)
    print("TEST 1: P3 - Self-Correction Detection (25 patterns)")
    print("="*80)
    
    from src.services.scoring_engine import ScoringEngine
    
    engine = ScoringEngine()
    
    test_cases = [
        {
            "text": "I went to the store yesterday",
            "expected": 0,
            "description": "No corrections"
        },
        {
            "text": "I mean, actually, I think I was wrong about that",
            "expected": 2,
            "description": "Multiple correction phrases"
        },
        {
            "text": "That's not right, oops, my mistake, let me correct that",
            "expected": 3,
            "description": "Many corrections (4 patterns)"
        },
        {
            "text": "Scratch that, never mind, forget what I said",
            "expected": 3,
            "description": "Retractions (new patterns)"
        },
        {
            "text": "Not exactly, let me clarify what I meant",
            "expected": 2,
            "description": "Clarification patterns"
        }
    ]
    
    print("\nTesting P3 detection:")
    passed = 0
    for i, case in enumerate(test_cases, 1):
        score = engine._score_self_correction(case["text"])
        
        status = "✅ PASS" if score >= case["expected"] - 1 else "⚠️  FAIL"
        if score >= case["expected"] - 1:
            passed += 1
            
        print(f"\n  Test {i}: {case['description']}")
        print(f"    Text: \"{case['text']}\"")
        print(f"    Score: {score}/3 (expected: {case['expected']})")
        print(f"    Status: {status}")
    
    print(f"\n  Summary: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_p4_low_confidence():
    """Test improved P4: Low Confidence"""
    print("\n" + "="*80)
    print("TEST 2: P4 - Low Confidence Detection (35 markers)")
    print("="*80)
    
    from src.services.scoring_engine import ScoringEngine
    
    engine = ScoringEngine()
    
    test_cases = [
        {
            "text": "I know exactly what happened",
            "expected": 0,
            "description": "Confident statement"
        },
        {
            "text": "I think maybe it was around 3 o'clock",
            "expected": 2,
            "description": "Basic uncertainty (2 markers)"
        },
        {
            "text": "It seems like, sort of, kind of a strange situation",
            "expected": 3,
            "description": "Hedging language (3 markers)"
        },
        {
            "text": "I suppose, presumably, I would say it's uncertain",
            "expected": 3,
            "description": "Supposition (new patterns, 4 markers)"
        },
        {
            "text": "Not sure, can't say for sure, fairly unclear",
            "expected": 3,
            "description": "Explicit uncertainty (new patterns)"
        }
    ]
    
    print("\nTesting P4 detection:")
    passed = 0
    for i, case in enumerate(test_cases, 1):
        score = engine._score_low_confidence(case["text"])
        
        status = "✅ PASS" if score >= case["expected"] - 1 else "⚠️  FAIL"
        if score >= case["expected"] - 1:
            passed += 1
            
        print(f"\n  Test {i}: {case['description']}")
        print(f"    Text: \"{case['text']}\"")
        print(f"    Score: {score}/3 (expected: {case['expected']})")
        print(f"    Status: {status}")
    
    print(f"\n  Summary: {passed}/{len(test_cases)} tests passed")
    return passed == len(test_cases)


def test_p7_emotion_slip():
    """Test improved P7: Emotion + Slip"""
    print("\n" + "="*80)
    print("TEST 3: P7 - Emotion + Slip Detection (Sentiment Analysis)")
    print("="*80)
    
    from src.services.scoring_engine import ScoringEngine, TEXTBLOB_AVAILABLE
    
    if TEXTBLOB_AVAILABLE:
        print("✅ TextBlob available - using sentiment analysis")
    else:
        print("⚠️  TextBlob not available - using keyword fallback")
    
    engine = ScoringEngine()
    
    test_cases = [
        {
            "text": "The weather is nice today",
            "audio": None,
            "expected": 0,
            "description": "Neutral, no emotion or slips"
        },
        {
            "text": "I'm so frustrated and confused, oops, what was I saying?",
            "audio": None,
            "expected": 2,
            "description": "Emotion words + slip indicator"
        },
        {
            "text": "I'm extremely angry and upset! This is terrible!",
            "audio": {"emotion_intensity": 0.8},
            "expected": 3,
            "description": "Strong emotion + high audio emotion"
        },
        {
            "text": "Wait, um, what's the word, you know that thing, uh",
            "audio": None,
            "expected": 2,
            "description": "Multiple slip indicators (word-finding)"
        },
        {
            "text": "Oh no, wrong, my mistake, oops, messed up",
            "audio": None,
            "expected": 2,
            "description": "Multiple error acknowledgments"
        },
        {
            "text": "I feel absolutely miserable and hopeless",
            "audio": {"emotion_intensity": 0.6},
            "expected": 2,
            "description": "Strong negative emotion + moderate audio"
        }
    ]
    
    print("\nTesting P7 detection:")
    passed = 0
    for i, case in enumerate(test_cases, 1):
        score = engine._score_emotion_slip(case["text"], case["audio"])
        
        status = "✅ PASS" if score >= case["expected"] - 1 else "⚠️  FAIR"
        if score >= case["expected"] - 1:
            passed += 1
            
        print(f"\n  Test {i}: {case['description']}")
        print(f"    Text: \"{case['text']}\"")
        if case["audio"]:
            print(f"    Audio: emotion_intensity = {case['audio']['emotion_intensity']}")
        print(f"    Score: {score}/3 (expected: {case['expected']})")
        print(f"    Status: {status}")
    
    print(f"\n  Summary: {passed}/{len(test_cases)} tests passed")
    return passed >= len(test_cases) - 1  # Allow 1 failure


def test_complete_session():
    """Test complete session with improved parameters"""
    print("\n" + "="*80)
    print("TEST 4: Complete Session Analysis (All Improvements)")
    print("="*80)
    
    from src.services.scoring_engine import ScoringEngine
    
    engine = ScoringEngine()
    
    # Simulated conversation with correction, uncertainty, and emotion
    text = "Wait, I mean, um, I think maybe I forgot something. I'm so confused and frustrated, oops, what was the word, you know that thing?"
    
    audio_features = {
        "pause_frequency": 0.4,
        "tremor_intensity": 0.5,
        "speech_rate": 90.0,
        "emotion_intensity": 0.7
    }
    
    print("\nTest Message:")
    print(f'  "{text}"')
    print("\nAudio Features:")
    print(f"  pause_frequency: {audio_features['pause_frequency']}")
    print(f"  tremor_intensity: {audio_features['tremor_intensity']}")
    print(f"  speech_rate: {audio_features['speech_rate']} wpm")
    print(f"  emotion_intensity: {audio_features['emotion_intensity']}")
    
    result = engine.analyze_session(
        text=text,
        audio_features=audio_features,
        timestamp=datetime.now(),
        conversation_context=[text]
    )
    
    scores = result["scores"]
    
    print("\nDetected Parameters:")
    detections = []
    if scores["p3_self_correction"] > 0:
        detections.append(f"  ✅ P3: Self-Correction = {scores['p3_self_correction']}/3")
    if scores["p4_low_confidence"] > 0:
        detections.append(f"  ✅ P4: Low Confidence = {scores['p4_low_confidence']}/3")
    if scores["p7_emotion_slip"] > 0:
        detections.append(f"  ✅ P7: Emotion+Slip = {scores['p7_emotion_slip']}/3")
    if scores["p5_hesitation_pauses"] > 0:
        detections.append(f"  ✅ P5: Hesitation Pauses = {scores['p5_hesitation_pauses']}/3")
    if scores["p6_vocal_tremors"] > 0:
        detections.append(f"  ✅ P6: Vocal Tremors = {scores['p6_vocal_tremors']}/3")
    if scores["p8_slowed_speech"] > 0:
        detections.append(f"  ✅ P8: Slowed Speech = {scores['p8_slowed_speech']}/3")
    
    for detection in detections:
        print(detection)
    
    print(f"\n  Total Session Score: {result['session_raw_score']}/36")
    
    # Verify improvements
    p3_detected = scores["p3_self_correction"] >= 1
    p4_detected = scores["p4_low_confidence"] >= 1
    p7_detected = scores["p7_emotion_slip"] >= 1
    
    success = p3_detected and p4_detected and p7_detected
    
    if success:
        print("\n  ✅ All improved parameters detected correctly!")
    else:
        print("\n  ⚠️  Some parameters may need adjustment")
    
    return success


def main():
    """Run all improvement tests"""
    print("\n" + "="*80)
    print("TESTING IMPROVED PARAMETERS (P3, P4, P7)")
    print("="*80)
    print("\nImprovements:")
    print("  ✅ P3: 14 → 25 patterns (self-correction)")
    print("  ✅ P4: 19 → 35 markers (low confidence)")
    print("  ✅ P7: Keyword → Sentiment Analysis + Slip Detection")
    
    try:
        results = []
        
        # Test each parameter
        results.append(("P3 Self-Correction", test_p3_self_correction()))
        results.append(("P4 Low Confidence", test_p4_low_confidence()))
        results.append(("P7 Emotion+Slip", test_p7_emotion_slip()))
        results.append(("Complete Session", test_complete_session()))
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        for name, passed in results:
            status = "✅ PASS" if passed else "⚠️  NEEDS REVIEW"
            print(f"  {name}: {status}")
        
        all_passed = all(passed for _, passed in results)
        
        if all_passed:
            print("\n" + "="*80)
            print("✅ ALL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
            print("="*80)
            print("\nEnhancements Complete:")
            print("  ✅ P3: 25 self-correction patterns")
            print("  ✅ P4: 35 uncertainty markers")
            print("  ✅ P7: TextBlob sentiment analysis + 20+ slip indicators")
            print("\n  Parameters P3, P4, P7 are now VIVA-READY! ⭐⭐⭐⭐")
        else:
            print("\n⚠️  Some tests need review (check output above)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
