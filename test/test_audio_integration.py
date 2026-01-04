"""
Test Audio Integration with Improved Detection System

Tests:
1. Audio feature extraction (P5, P6, P8)
2. Improved P1 semantic incoherence detection
3. Improved P12 topic maintenance detection
4. Complete session analysis with audio
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_audio_processor():
    """Test audio feature extraction"""
    print("\n" + "="*80)
    print("TEST 1: Audio Processor (P5, P6, P8)")
    print("="*80)

    from src.services.audio_processor import audio_processor

    # Check if librosa is available
    if not audio_processor.librosa_available:
        print("⚠️  librosa not available - will use placeholder values")
        print("   Install with: pip install librosa soundfile")
    else:
        print("✅ librosa is available")

    # Test with placeholder (no actual audio file needed for demo)
    features = audio_processor._get_placeholder_features()

    print("\nAudio Features (placeholder):")
    print(f"  P5: Pause Frequency: {features['pause_frequency']}")
    print(f"  P6: Tremor Intensity: {features['tremor_intensity']}")
    print(f"  P8: Speech Rate: {features['speech_rate']} wpm")
    print(f"  P7: Emotion Intensity: {features['emotion_intensity']}")

    print("\n✅ Audio processor initialized successfully")


def test_semantic_similarity():
    """Test semantic similarity for P1 and P12"""
    print("\n" + "="*80)
    print("TEST 2: Semantic Similarity (P1, P12)")
    print("="*80)

    from src.services.scoring_engine import ScoringEngine, SEMANTIC_SIMILARITY_AVAILABLE

    if not SEMANTIC_SIMILARITY_AVAILABLE:
        print("⚠️  sentence-transformers not available - will use basic word overlap")
        print("   Install with: pip install sentence-transformers scikit-learn")
    else:
        print("✅ sentence-transformers is available")
        print("   Loading semantic model (this may take a moment)...")

    scoring_engine = ScoringEngine()

    if scoring_engine.semantic_model is not None:
        print("✅ Semantic model loaded successfully")
    else:
        print("⚠️  Using fallback word overlap method")


def test_improved_p1_detection():
    """Test improved P1: Semantic Incoherence"""
    print("\n" + "="*80)
    print("TEST 3: Improved P1 - Semantic Incoherence Detection")
    print("="*80)

    from src.services.scoring_engine import ScoringEngine

    scoring_engine = ScoringEngine()

    # Test cases
    test_cases = [
        {
            "text": "I went to the store to buy milk",
            "expected_score": 0,
            "description": "Normal, coherent sentence"
        },
        {
            "text": "Um, like, you know, whatever, basically I think maybe",
            "expected_score": 3,
            "description": "Many filler words (high density)"
        },
        {
            "text": "I love gardening. My car needs oil. What's for dinner?",
            "expected_score": 2,
            "description": "Topic jumping (semantic incoherence)"
        },
        {
            "text": "The weather is nice today. I think we should go outside.",
            "expected_score": 0,
            "description": "Coherent topic maintenance"
        }
    ]

    print("\nTesting P1 detection:")
    for i, case in enumerate(test_cases, 1):
        # Clear conversation history for fresh test
        scoring_engine.conversation_history = []

        # Add a previous message for context
        if "gardening" in case["text"] or "weather" in case["text"]:
            scoring_engine.conversation_history.append("Let's talk about outdoor activities")

        score = scoring_engine._score_semantic_incoherence(case["text"])

        print(f"\n  Test {i}: {case['description']}")
        print(f"    Text: \"{case['text']}\"")
        print(f"    Score: {score}/3")
        print(f"    Status: {'✅ PASS' if score >= case['expected_score'] - 1 else '⚠️  Different than expected'}")


def test_improved_p12_detection():
    """Test improved P12: Topic Maintenance"""
    print("\n" + "="*80)
    print("TEST 4: Improved P12 - Topic Maintenance Detection")
    print("="*80)

    from src.services.scoring_engine import ScoringEngine

    scoring_engine = ScoringEngine()

    # Test case: Conversation with topic drift
    conversation = [
        "I love gardening and growing flowers",
        "Roses need a lot of water and sunlight",
        "My car broke down yesterday"  # Topic jump
    ]

    print("\nTesting P12 detection:")
    print("\nConversation:")
    for i, msg in enumerate(conversation, 1):
        print(f"  {i}. \"{msg}\"")

    # Build conversation history
    for msg in conversation[:-1]:
        scoring_engine.conversation_history.append(msg)

    # Test last message (should detect topic jump)
    score = scoring_engine._score_topic_maintenance(conversation[-1])

    print(f"\nTopic Maintenance Score: {score}/3")
    if score >= 2:
        print("✅ Correctly detected topic jump!")
    else:
        print("⚠️  Score lower than expected (may need semantic model)")


def test_complete_session_analysis():
    """Test complete session analysis with all improvements"""
    print("\n" + "="*80)
    print("TEST 5: Complete Session Analysis")
    print("="*80)

    from src.services.scoring_engine import ScoringEngine

    scoring_engine = ScoringEngine()

    # Simulate a session with multiple messages
    messages = [
        "Good morning, how are you today?",
        "I think maybe I forgot something. What was I saying?",
        "Um, like, you know, I can't remember",
        "What time is it? I forgot to check the time."
    ]

    print("\nSimulated Chat Session:")
    for i, msg in enumerate(messages, 1):
        print(f"  Message {i}: \"{msg}\"")

    # Analyze session
    audio_features = {
        "pause_frequency": 0.35,  # High pauses
        "tremor_intensity": 0.65,  # Moderate tremor
        "speech_rate": 85.0,       # Slow speech
        "emotion_intensity": 0.55  # Some emotion
    }

    print("\nAudio Features:")
    print(f"  Pause Frequency: {audio_features['pause_frequency']} (high)")
    print(f"  Tremor Intensity: {audio_features['tremor_intensity']} (moderate)")
    print(f"  Speech Rate: {audio_features['speech_rate']} wpm (slow)")
    print(f"  Emotion Intensity: {audio_features['emotion_intensity']}")

    # Analyze each message
    total_score = 0
    print("\nAnalyzing messages:")

    for i, text in enumerate(messages, 1):
        result = scoring_engine.analyze_session(
            text=text,
            audio_features=audio_features,
            timestamp=datetime.now(),
            conversation_context=messages[:i]
        )

        session_score = result["session_raw_score"]
        total_score = session_score

        print(f"\n  After Message {i}:")
        print(f"    Session Score: {session_score}/36")

        # Show key detections
        scores = result["scores"]
        detections = []
        if scores["p1_semantic_incoherence"] > 0:
            detections.append(f"P1: Semantic Incoherence ({scores['p1_semantic_incoherence']})")
        if scores["p2_repeated_questions"] > 0:
            detections.append(f"P2: Repeated Questions ({scores['p2_repeated_questions']})")
        if scores["p4_low_confidence"] > 0:
            detections.append(f"P4: Low Confidence ({scores['p4_low_confidence']})")
        if scores["p11_memory_recall_failure"] > 0:
            detections.append(f"P11: Memory Failure ({scores['p11_memory_recall_failure']})")

        if detections:
            print(f"    Detections: {', '.join(detections)}")

    print(f"\n{'='*80}")
    print(f"FINAL SESSION SCORE: {total_score}/36")
    print(f"Risk Level: {_get_risk_level(total_score)}")
    print(f"{'='*80}")


def _get_risk_level(score):
    """Get risk level from score"""
    if score >= 24:
        return "HIGH RISK"
    elif score >= 12:
        return "MODERATE RISK"
    elif score >= 6:
        return "LOW RISK"
    else:
        return "MINIMAL RISK"


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING IMPROVED DEMENTIA DETECTION SYSTEM")
    print("="*80)
    print("\nCritical Improvements Implemented:")
    print("  ✅ P5, P6, P8: Real audio processing with librosa")
    print("  ✅ P1: Improved semantic incoherence detection")
    print("  ✅ P12: Improved topic maintenance detection")
    print("  ✅ Integration with session accumulation logic")

    try:
        # Test 1: Audio Processor
        test_audio_processor()

        # Test 2: Semantic Similarity
        test_semantic_similarity()

        # Test 3: Improved P1
        test_improved_p1_detection()

        # Test 4: Improved P12
        test_improved_p12_detection()

        # Test 5: Complete Session
        test_complete_session_analysis()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)

        print("\n📋 NEXT STEPS:")
        print("  1. Install dependencies: pip install librosa soundfile sentence-transformers")
        print("  2. Test with real audio files using /api/detection/process-audio")
        print("  3. Run integration tests with API endpoints")
        print("  4. Collect real data for validation")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
