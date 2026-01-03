"""
Test Script for Weekly Report Generator

Demonstrates generating weekly cognitive risk reports
for dementia patients using the reminder system data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.reminder_system.behavior_tracker import BehaviorTracker
from src.features.reminder_system.weekly_report_generator import WeeklyReportGenerator
from src.features.reminder_system.reminder_models import (
    ReminderInteraction, InteractionType, CaregiverAlert
)


def generate_sample_data():
    """Generate sample interaction data for testing."""
    behavior_tracker = BehaviorTracker()
    
    # Sample user
    user_id = "patient_demo_001"
    
    print("Generating sample interaction data...")
    
    # Simulate 7 days of interactions
    base_date = datetime.now() - timedelta(days=7)
    
    for day in range(7):
        day_date = base_date + timedelta(days=day)
        
        # Morning medication (8 AM)
        interaction1 = ReminderInteraction(
            user_id=user_id,
            reminder_id=f"rem_morning_{day}",
            reminder_category="medication",
            interaction_time=day_date.replace(hour=8, minute=5),
            interaction_type=InteractionType.CONFIRMED if day < 5 else InteractionType.CONFUSED,
            cognitive_risk_score=0.15 + (day * 0.05),  # Increasing risk
            confusion_detected=day >= 5,
            memory_issue_detected=day >= 6,
            response_time_seconds=30 + (day * 10)
        )
        behavior_tracker.log_interaction(interaction1)
        
        # Lunch reminder (12 PM)
        interaction2 = ReminderInteraction(
            user_id=user_id,
            reminder_id=f"rem_lunch_{day}",
            reminder_category="meal",
            interaction_time=day_date.replace(hour=12, minute=10),
            interaction_type=InteractionType.CONFIRMED if day < 6 else InteractionType.DELAYED,
            cognitive_risk_score=0.10 + (day * 0.04),
            confusion_detected=False,
            memory_issue_detected=False,
            response_time_seconds=20 + (day * 5)
        )
        behavior_tracker.log_interaction(interaction2)
        
        # Evening medication (8 PM)
        interaction3 = ReminderInteraction(
            user_id=user_id,
            reminder_id=f"rem_evening_{day}",
            reminder_category="medication",
            interaction_time=day_date.replace(hour=20, minute=15),
            interaction_type=InteractionType.IGNORED if day >= 5 else InteractionType.CONFIRMED,
            cognitive_risk_score=0.25 + (day * 0.08),  # Higher risk in evening
            confusion_detected=day >= 4,
            memory_issue_detected=day >= 5,
            response_time_seconds=60 + (day * 15)
        )
        behavior_tracker.log_interaction(interaction3)
    
    print(f"✓ Generated {len(behavior_tracker.interaction_cache[f'{user_id}_*'])} interactions")
    
    return behavior_tracker, user_id


def test_weekly_report():
    """Test weekly report generation."""
    
    print("\n" + "="*70)
    print("WEEKLY COGNITIVE RISK REPORT - TEST")
    print("="*70 + "\n")
    
    # Generate sample data
    behavior_tracker, user_id = generate_sample_data()
    
    # Create report generator
    report_generator = WeeklyReportGenerator(behavior_tracker)
    
    # Generate report
    print("Generating weekly report...\n")
    report = report_generator.generate_weekly_report(user_id=user_id)
    
    # Display report
    print("="*70)
    print("WEEKLY COGNITIVE RISK MONITORING REPORT")
    print("="*70)
    print(f"\nPatient ID: {report.user_id}")
    print(f"Report Period: {report.report_period_start[:10]} to {report.report_period_end[:10]}")
    print(f"Generated: {report.generated_at[:19]}")
    
    print("\n" + "-"*70)
    print("OVERALL STATISTICS")
    print("-"*70)
    print(f"Total Reminders: {report.total_reminders}")
    print(f"Completed: {report.completed_reminders}")
    print(f"Missed: {report.missed_reminders}")
    print(f"Completion Rate: {report.completion_rate:.1%}")
    
    print("\n" + "-"*70)
    print("COGNITIVE HEALTH METRICS")
    print("-"*70)
    print(f"Risk Level: {report.risk_level.upper()}")
    print(f"Average Cognitive Risk: {report.avg_cognitive_risk_score:.3f}")
    print(f"Peak Risk Score: {report.peak_cognitive_risk_score:.3f}")
    print(f"Lowest Risk Score: {report.lowest_cognitive_risk_score:.3f}")
    print(f"Risk Trend: {report.risk_trend.upper()}")
    
    print("\n" + "-"*70)
    print("INTERACTION PATTERNS")
    print("-"*70)
    print(f"Confirmed: {report.confirmed_count}")
    print(f"Confused: {report.confusion_count}")
    print(f"Memory Issues: {report.memory_issue_count}")
    print(f"Delayed: {report.delayed_count}")
    print(f"Ignored: {report.ignored_count}")
    
    print("\n" + "-"*70)
    print("DAILY BREAKDOWN")
    print("-"*70)
    print(f"{'Date':<12} {'Avg Risk':<10} {'Confusion':<12} {'Completion':<12} {'Alerts':<8}")
    print("-"*70)
    for daily in report.daily_summaries:
        print(f"{daily.date:<12} {daily.avg_cognitive_risk:<10.3f} "
              f"{daily.confusion_count:<12} {daily.completion_rate:<11.1%} {daily.alert_count:<8}")
    
    print("\n" + "-"*70)
    print("TIME ANALYSIS")
    print("-"*70)
    best_hours = ", ".join(f"{h}:00" for h in report.best_response_hours) if report.best_response_hours else "N/A"
    worst_hours = ", ".join(f"{h}:00" for h in report.worst_response_hours) if report.worst_response_hours else "N/A"
    print(f"Best Response Hours: {best_hours}")
    print(f"Worst Response Hours: {worst_hours}")
    print(f"Avg Response Time: {report.avg_response_time_seconds:.1f} seconds")
    
    print("\n" + "-"*70)
    print("CATEGORY PERFORMANCE")
    print("-"*70)
    if report.category_breakdown:
        for category, stats in report.category_breakdown.items():
            print(f"\n{category.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Completed: {stats['completed']}")
            print(f"  Completion Rate: {stats['completion_rate']:.1%}")
            print(f"  Avg Risk: {stats['avg_risk']:.3f}")
    
    print("\n" + "-"*70)
    print("CAREGIVER ALERTS")
    print("-"*70)
    print(f"Total Alerts: {report.total_alerts}")
    print(f"Critical: {report.critical_alerts}")
    print(f"High Priority: {report.high_priority_alerts}")
    print(f"Unresolved: {report.unresolved_alerts}")
    
    print("\n" + "-"*70)
    print("ASSESSMENT")
    print("-"*70)
    print(f"Intervention Needed: {'YES ⚠️' if report.intervention_needed else 'NO ✓'}")
    print(f"Escalation Required: {'YES 🚨' if report.escalation_required else 'NO ✓'}")
    
    print("\n" + "-"*70)
    print("RECOMMENDATIONS")
    print("-"*70)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    
    if report.previous_week_avg_risk is not None:
        print("\n" + "-"*70)
        print("WEEK-OVER-WEEK COMPARISON")
        print("-"*70)
        print(f"Previous Week Avg Risk: {report.previous_week_avg_risk:.3f}")
        change = ((report.avg_cognitive_risk_score - report.previous_week_avg_risk) 
                  / report.previous_week_avg_risk * 100)
        arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"Change: {arrow} {change:+.1f}%")
    
    print("\n" + "="*70 + "\n")
    
    # Export to JSON
    output_file = f"weekly_report_{user_id}_{datetime.now().strftime('%Y%m%d')}.json"
    report_generator.export_report_to_json(report, output_file)
    print(f"✓ Report exported to: {output_file}\n")
    
    return report


def test_api_endpoint_format():
    """Show example API response format."""
    
    print("\n" + "="*70)
    print("API ENDPOINT RESPONSE FORMAT")
    print("="*70)
    
    print("\nEndpoint: GET /api/reminders/reports/weekly/{user_id}")
    print("\nExample Response:")
    
    example_response = {
        "status": "success",
        "report": {
            "user_id": "patient_123",
            "report_period_start": "2026-01-01T00:00:00",
            "report_period_end": "2026-01-08T00:00:00",
            "risk_level": "moderate",
            "avg_cognitive_risk_score": 0.45,
            "risk_trend": "declining",
            "completion_rate": 0.71,
            "confusion_count": 5,
            "total_alerts": 3,
            "intervention_needed": True,
            "recommendations": [
                "⚠️ Elevated cognitive risk. Schedule follow-up.",
                "📉 Performance declining. Increase monitoring.",
                "⏰ Poor performance at 20:00, 21:00. Reschedule."
            ]
        }
    }
    
    print(json.dumps(example_response, indent=2))
    
    print("\n" + "="*70)
    print("SUMMARY ENDPOINT")
    print("="*70)
    
    print("\nEndpoint: GET /api/reminders/reports/weekly/{user_id}/summary")
    print("\nExample Response:")
    
    summary_response = {
        "status": "success",
        "summary": {
            "user_id": "patient_123",
            "period": "2026-01-01T00:00:00 to 2026-01-08T00:00:00",
            "risk_level": "moderate",
            "avg_cognitive_risk": 0.45,
            "risk_trend": "declining",
            "completion_rate": 0.71,
            "total_alerts": 3,
            "intervention_needed": True,
            "top_recommendations": [
                "⚠️ Elevated cognitive risk. Schedule follow-up.",
                "📉 Performance declining. Increase monitoring.",
                "⏰ Poor performance at 20:00, 21:00. Reschedule."
            ]
        }
    }
    
    print(json.dumps(summary_response, indent=2))
    print()


if __name__ == "__main__":
    print("\n🏥 Weekly Report Generator Test\n")
    
    # Run tests
    report = test_weekly_report()
    test_api_endpoint_format()
    
    print("✅ Test completed successfully!\n")
