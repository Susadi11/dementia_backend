"""
Test Reminder System Endpoints

Quick test script to verify all reminder endpoints are working correctly.
Run this to check your frontend integration endpoints.
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def print_test(name, passed, response=None):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status} - {name}")
    if not passed and response:
        print(f"  Error: {response.text}")
    print()

def test_health():
    """Test reminder system health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/reminders/health")
        passed = response.status_code == 200
        print_test("Health Check", passed, response)
        if passed:
            print(f"  Response: {response.json()}\n")
        return passed
    except Exception as e:
        print_test("Health Check", False)
        print(f"  Exception: {e}\n")
        return False

def test_create_reminder():
    """Test creating a reminder."""
    try:
        reminder_data = {
            "id": f"reminder_{datetime.now().timestamp()}",
            "user_id": "patient_001",
            "title": "Take Morning Medication",
            "description": "Blood pressure pills - 2 tablets",
            "scheduled_time": (datetime.now() + timedelta(hours=1)).isoformat(),
            "priority": "high",
            "category": "medication",
            "recurrence": "daily",
            "caregiver_ids": ["caregiver_001"]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/reminders/create",
            json=reminder_data
        )
        passed = response.status_code == 201
        print_test("Create Reminder", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed, reminder_data["id"]
    except Exception as e:
        print_test("Create Reminder", False)
        print(f"  Exception: {e}\n")
        return False, None

def test_natural_language_reminder():
    """Test creating reminder from natural language."""
    try:
        command_data = {
            "user_id": "patient_001",
            "command_text": "Remind me to take my blood pressure medicine every morning at 8",
            "audio_path": None,
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{BASE_URL}/api/reminders/natural-language",
            json=command_data
        )
        passed = response.status_code == 200
        print_test("Natural Language Reminder", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Natural Language Reminder", False)
        print(f"  Exception: {e}\n")
        return False

def test_get_user_reminders():
    """Test getting user reminders."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/reminders/user/patient_001?status_filter=active"
        )
        passed = response.status_code == 200
        print_test("Get User Reminders", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Get User Reminders", False)
        print(f"  Exception: {e}\n")
        return False

def test_respond_to_reminder():
    """Test processing response to reminder."""
    try:
        response_data = {
            "reminder_id": "reminder_001",
            "user_id": "patient_001",
            "response_text": "Yes, I took my medication",
            "audio_path": None,
            "response_timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{BASE_URL}/api/reminders/respond",
            json=response_data
        )
        passed = response.status_code == 200
        print_test("Process Reminder Response", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Process Reminder Response", False)
        print(f"  Exception: {e}\n")
        return False

def test_behavior_analysis():
    """Test behavior pattern analysis."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/reminders/behavior/patient_001?days=30"
        )
        passed = response.status_code == 200
        print_test("Behavior Analysis", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Behavior Analysis", False)
        print(f"  Exception: {e}\n")
        return False

def test_dashboard_analytics():
    """Test dashboard analytics."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/reminders/analytics/dashboard/patient_001?days=7"
        )
        passed = response.status_code == 200
        print_test("Dashboard Analytics", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Dashboard Analytics", False)
        print(f"  Exception: {e}\n")
        return False

def test_caregiver_alerts():
    """Test getting caregiver alerts."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/reminders/caregiver/alerts/caregiver_001?active_only=true"
        )
        passed = response.status_code == 200
        print_test("Caregiver Alerts", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Caregiver Alerts", False)
        print(f"  Exception: {e}\n")
        return False

def test_weekly_report():
    """Test weekly report generation."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/reminders/reports/weekly/patient_001?format=json"
        )
        passed = response.status_code == 200
        print_test("Weekly Report", passed, response)
        if passed:
            data = response.json()
            print(f"  Report Period: {data.get('report', {}).get('reporting_period')}")
            print(f"  Risk Level: {data.get('report', {}).get('overall_risk_level')}\n")
        return passed
    except Exception as e:
        print_test("Weekly Report", False)
        print(f"  Exception: {e}\n")
        return False

def test_weekly_report_summary():
    """Test weekly report summary."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/reminders/reports/weekly/patient_001/summary"
        )
        passed = response.status_code == 200
        print_test("Weekly Report Summary", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Weekly Report Summary", False)
        print(f"  Exception: {e}\n")
        return False

def test_snooze_reminder(reminder_id):
    """Test snoozing a reminder."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/reminders/snooze/{reminder_id}?delay_minutes=15"
        )
        passed = response.status_code == 200
        print_test("Snooze Reminder", passed, response)
        if passed:
            print(f"  Response: {json.dumps(response.json(), indent=2)}\n")
        return passed
    except Exception as e:
        print_test("Snooze Reminder", False)
        print(f"  Exception: {e}\n")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("REMINDER SYSTEM ENDPOINT TESTS")
    print("="*80)
    print()
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health()))
    
    # Test 2: Create Reminder
    create_passed, reminder_id = test_create_reminder()
    results.append(("Create Reminder", create_passed))
    
    # Test 3: Natural Language
    results.append(("Natural Language", test_natural_language_reminder()))
    
    # Test 4: Get User Reminders
    results.append(("Get Reminders", test_get_user_reminders()))
    
    # Test 5: Process Response
    results.append(("Process Response", test_respond_to_reminder()))
    
    # Test 6: Behavior Analysis
    results.append(("Behavior Analysis", test_behavior_analysis()))
    
    # Test 7: Dashboard Analytics
    results.append(("Dashboard Analytics", test_dashboard_analytics()))
    
    # Test 8: Caregiver Alerts
    results.append(("Caregiver Alerts", test_caregiver_alerts()))
    
    # Test 9: Weekly Report
    results.append(("Weekly Report", test_weekly_report()))
    
    # Test 10: Weekly Report Summary
    results.append(("Report Summary", test_weekly_report_summary()))
    
    # Test 11: Snooze Reminder
    if reminder_id:
        results.append(("Snooze Reminder", test_snooze_reminder(reminder_id)))
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print()
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print()
    if passed == total:
        print("🎉 All tests passed! Your endpoints are ready for frontend integration.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    print()


if __name__ == "__main__":
    main()
