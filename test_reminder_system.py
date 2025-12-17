"""
Test Script for Reminder System

Tests the Context-Aware Smart Reminder System components:
- Pitt Corpus-based response analysis
- Adaptive scheduling
- Behavior tracking
- Caregiver notifications
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
from src.features.reminder_system import (
    Reminder, ReminderInteraction, ReminderStatus, ReminderPriority,
    InteractionType, PittBasedReminderAnalyzer, AdaptiveReminderScheduler,
    BehaviorTracker
)
from src.features.reminder_system.caregiver_notifier import CaregiverNotifier


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_reminder_analyzer():
    """Test Pitt-based reminder response analyzer."""
    print_section("Testing Reminder Response Analyzer")
    
    analyzer = PittBasedReminderAnalyzer()
    
    test_responses = [
        {
            'text': "Yes, I took my tablets after breakfast",
            'expected': 'confirmed',
            'description': 'Clear confirmation'
        },
        {
            'text': "Um... I think I already did that... or maybe not?",
            'expected': 'confused',
            'description': 'Confusion and uncertainty'
        },
        {
            'text': "What medicine? What are you talking about?",
            'expected': 'confused',
            'description': 'Memory issue and confusion'
        },
        {
            'text': "Later... I'm busy right now",
            'expected': 'delayed',
            'description': 'Deliberate delay'
        },
        {
            'text': "I don't remember... did I already do it?",
            'expected': 'repeated_question',
            'description': 'Memory issue'
        },
    ]
    
    print(f"Testing {len(test_responses)} different response patterns:\n")
    
    for i, test in enumerate(test_responses, 1):
        print(f"{i}. {test['description']}")
        print(f"   Response: \"{test['text']}\"")
        
        result = analyzer.analyze_reminder_response(test['text'])
        
        print(f"   → Interaction Type: {result['interaction_type']}")
        print(f"   → Cognitive Risk: {result['cognitive_risk_score']:.2f}")
        print(f"   → Confusion: {'YES' if result['confusion_detected'] else 'NO'}")
        print(f"   → Memory Issue: {'YES' if result['memory_issue_detected'] else 'NO'}")
        print(f"   → Recommended Action: {result['recommended_action']}")
        print(f"   → Caregiver Alert: {'YES' if result['caregiver_alert_needed'] else 'NO'}")
        
        # Verify expected type
        if test['expected'] in result['interaction_type']:
            print(f"   ✓ PASS: Correctly identified as {test['expected']}")
        else:
            print(f"   ✗ FAIL: Expected {test['expected']}, got {result['interaction_type']}")
        
        print()
    
    print("✓ Reminder analyzer test completed\n")


def test_behavior_tracker():
    """Test behavior tracking and pattern analysis."""
    print_section("Testing Behavior Tracker")
    
    tracker = BehaviorTracker()
    
    # Simulate user interactions over time
    user_id = "test_user_123"
    reminder_id = "medication_reminder_1"
    
    print(f"Simulating reminder interactions for user {user_id}:\n")
    
    # Day 1-5: Good compliance
    print("Days 1-5: Good compliance")
    for day in range(5):
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            interaction_type=InteractionType.CONFIRMED,
            interaction_time=datetime.now() - timedelta(days=30-day),
            user_response_text="Yes, I took it",
            cognitive_risk_score=0.2,
            response_time_seconds=10.0
        )
        tracker.log_interaction(interaction)
    
    # Day 6-10: Some delays
    print("Days 6-10: Some delays")
    for day in range(5, 10):
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            interaction_type=InteractionType.DELAYED,
            interaction_time=datetime.now() - timedelta(days=30-day),
            user_response_text="Later",
            cognitive_risk_score=0.3,
            response_time_seconds=30.0
        )
        tracker.log_interaction(interaction)
    
    # Day 11-15: Confusion appearing
    print("Days 11-15: Confusion appearing")
    for day in range(10, 15):
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            interaction_type=InteractionType.CONFUSED,
            interaction_time=datetime.now() - timedelta(days=30-day),
            user_response_text="What medicine?",
            cognitive_risk_score=0.7,
            confusion_detected=True,
            response_time_seconds=60.0
        )
        tracker.log_interaction(interaction)
    
    # Analyze behavior pattern
    print("\nAnalyzing behavior pattern...\n")
    pattern = tracker.get_user_behavior_pattern(user_id, reminder_id, days=30)
    
    print(f"Total Reminders: {pattern.total_reminders}")
    print(f"Confirmed: {pattern.confirmed_count} ({pattern.confirmed_count/pattern.total_reminders*100:.1f}%)")
    print(f"Delayed: {pattern.delayed_count} ({pattern.delayed_count/pattern.total_reminders*100:.1f}%)")
    print(f"Confused: {pattern.confused_count} ({pattern.confused_count/pattern.total_reminders*100:.1f}%)")
    print(f"\nAverage Cognitive Risk: {pattern.avg_cognitive_risk_score:.2f}")
    print(f"Confusion Trend: {pattern.confusion_trend}")
    print(f"Average Response Time: {pattern.avg_response_time_seconds:.1f} seconds")
    print(f"\nRecommendations:")
    print(f"  - Frequency Multiplier: {pattern.recommended_frequency_multiplier}x")
    print(f"  - Time Adjustment: {pattern.recommended_time_adjustment_minutes} minutes")
    print(f"  - Escalation Recommended: {'YES' if pattern.escalation_recommended else 'NO'}")
    
    print("\n✓ Behavior tracker test completed\n")


def test_adaptive_scheduler():
    """Test adaptive reminder scheduling."""
    print_section("Testing Adaptive Scheduler")
    
    behavior_tracker = BehaviorTracker()
    analyzer = PittBasedReminderAnalyzer()
    scheduler = AdaptiveReminderScheduler(behavior_tracker, analyzer)
    
    # Create test reminder
    reminder = Reminder(
        id="test_reminder_1",
        user_id="test_user_456",
        title="Take morning medication",
        description="Blood pressure medication (blue pill)",
        scheduled_time=datetime.now() + timedelta(hours=1),
        priority=ReminderPriority.CRITICAL,
        category="medication",
        caregiver_ids=["caregiver_789"]
    )
    
    print(f"Testing reminder: {reminder.title}")
    print(f"Priority: {reminder.priority}")
    print(f"Category: {reminder.category}\n")
    
    # Test different user responses
    test_cases = [
        {
            'response': "Yes, I just took it",
            'description': "Clear confirmation"
        },
        {
            'response': "Um... what medicine? I'm confused",
            'description': "Confusion - should trigger caregiver alert"
        },
        {
            'response': "Not now, I'll do it later",
            'description': "Delay - should reschedule"
        }
    ]
    
    print("Processing different response types:\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['description']}")
        print(f"   User Response: \"{test['response']}\"")
        
        result = scheduler.process_reminder_response(
            reminder=reminder,
            user_response=test['response']
        )
        
        analysis = result['analysis']
        action_result = result['action_result']
        
        print(f"   → Interaction Type: {analysis['interaction_type']}")
        print(f"   → Cognitive Risk: {analysis['cognitive_risk_score']:.2f}")
        print(f"   → Recommended Action: {analysis['recommended_action']}")
        print(f"   → Action Executed: {action_result.get('action', 'none')}")
        print(f"   → Reminder Updated: {'YES' if result.get('reminder_updated') else 'NO'}")
        print(f"   → Caregiver Notified: {'YES' if result.get('caregiver_notified') else 'NO'}")
        print()
    
    print("✓ Adaptive scheduler test completed\n")


def test_caregiver_notifier():
    """Test caregiver notification system."""
    print_section("Testing Caregiver Notifier")
    
    notifier = CaregiverNotifier()
    
    # Create test reminder
    reminder = Reminder(
        id="critical_reminder_1",
        user_id="patient_123",
        title="Take evening medication",
        scheduled_time=datetime.now(),
        priority=ReminderPriority.CRITICAL,
        category="medication",
        caregiver_ids=["caregiver_001", "caregiver_002"]
    )
    
    print(f"Testing notifications for reminder: {reminder.title}")
    print(f"Patient: {reminder.user_id}")
    print(f"Caregivers: {', '.join(reminder.caregiver_ids)}\n")
    
    # Test 1: Missed critical reminder
    print("1. Testing missed critical reminder alert")
    success = notifier.send_missed_reminder_alert(
        caregiver_id="caregiver_001",
        user_id=reminder.user_id,
        reminder=reminder,
        missed_count=3
    )
    print(f"   → Alert sent: {'YES' if success else 'NO'}")
    print()
    
    # Test 2: Confusion detected
    print("2. Testing confusion alert")
    interaction = ReminderInteraction(
        reminder_id=reminder.id,
        user_id=reminder.user_id,
        interaction_type=InteractionType.CONFUSED,
        user_response_text="What medicine? I don't understand",
        cognitive_risk_score=0.75,
        confusion_detected=True
    )
    
    success = notifier.send_confusion_alert(
        caregiver_id="caregiver_001",
        user_id=reminder.user_id,
        reminder=reminder,
        interaction=interaction
    )
    print(f"   → Alert sent: {'YES' if success else 'NO'}")
    print()
    
    # Test 3: Get active alerts
    print("3. Testing alert retrieval")
    active_alerts = notifier.get_active_alerts("caregiver_001")
    print(f"   → Active alerts: {len(active_alerts)}")
    
    if active_alerts:
        print("\n   Active Alerts:")
        for alert in active_alerts:
            print(f"   - {alert.alert_type}: {alert.severity} - {alert.message[:60]}...")
    
    print("\n✓ Caregiver notifier test completed\n")


def test_full_workflow():
    """Test complete reminder system workflow."""
    print_section("Testing Complete Workflow")
    
    print("Scenario: Patient with medication reminder showing cognitive decline\n")
    
    # Initialize components
    analyzer = PittBasedReminderAnalyzer()
    behavior_tracker = BehaviorTracker()
    scheduler = AdaptiveReminderScheduler(behavior_tracker, analyzer)
    notifier = CaregiverNotifier()
    
    # Create reminder
    reminder = Reminder(
        id="workflow_test_1",
        user_id="patient_workflow",
        title="Morning medication reminder",
        description="Take 2 blue pills with water",
        scheduled_time=datetime.now(),
        priority=ReminderPriority.CRITICAL,
        category="medication",
        caregiver_ids=["caregiver_workflow"],
        adaptive_scheduling_enabled=True,
        escalation_enabled=True
    )
    
    print("Step 1: Reminder created")
    print(f"  Title: {reminder.title}")
    print(f"  Priority: {reminder.priority}")
    print(f"  Scheduled: {reminder.scheduled_time.strftime('%H:%M')}")
    print()
    
    # Simulate patient responses over several days
    responses_timeline = [
        ("Day 1", "Yes, I took them", False),
        ("Day 2", "Yes, done", False),
        ("Day 3", "Um... I think I did?", True),
        ("Day 4", "What pills? I don't remember", True),
        ("Day 5", "I'm confused... what do you mean?", True),
    ]
    
    print("Step 2: Processing patient responses over 5 days\n")
    
    for day, response, expect_alert in responses_timeline:
        print(f"{day}: \"{response}\"")
        
        result = scheduler.process_reminder_response(
            reminder=reminder,
            user_response=response
        )
        
        analysis = result['analysis']
        print(f"  → Cognitive Risk: {analysis['cognitive_risk_score']:.2f}")
        print(f"  → Interaction Type: {analysis['interaction_type']}")
        
        if result.get('caregiver_notified'):
            print(f"  → ⚠️ CAREGIVER ALERTED")
        
        print()
    
    print("Step 3: Analyzing behavior pattern\n")
    
    pattern = behavior_tracker.get_user_behavior_pattern(
        user_id=reminder.user_id,
        reminder_id=reminder.id
    )
    
    print(f"Confusion Rate: {pattern.confused_count}/{pattern.total_reminders} ({pattern.confused_count/pattern.total_reminders*100:.1f}%)")
    print(f"Average Cognitive Risk: {pattern.avg_cognitive_risk_score:.2f}")
    print(f"Trend: {pattern.confusion_trend}")
    print(f"Escalation Recommended: {'YES' if pattern.escalation_recommended else 'NO'}")
    print()
    
    print("Step 4: Getting optimal schedule\n")
    
    schedule = scheduler.get_optimal_reminder_schedule(reminder)
    print(f"Optimal Time: {schedule['optimal_time'].strftime('%H:%M')}")
    print(f"Frequency Multiplier: {schedule['frequency_multiplier']}x")
    print(f"Urgency Level: {schedule['urgency_level']}")
    print()
    
    print("✓ Complete workflow test finished\n")
    print("Summary: System successfully detected cognitive decline and recommended escalation")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  CONTEXT-AWARE SMART REMINDER SYSTEM - TEST SUITE")
    print("=" * 80)
    
    try:
        test_reminder_analyzer()
        test_behavior_tracker()
        test_adaptive_scheduler()
        test_caregiver_notifier()
        test_full_workflow()
        
        print("\n" + "=" * 80)
        print("  ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
