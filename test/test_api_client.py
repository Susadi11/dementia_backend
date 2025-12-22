"""
Test API Client for Reminder System

Tests the REST API endpoints without external dependencies.
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all reminder system API endpoints."""
    
    print("🧪 Testing Context-Aware Smart Reminder System API")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        print("\n1. Testing Health Check")
        response = requests.get(f"{BASE_URL}/api/reminders/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"   Status: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
        
        # Test 2: Create reminder
        print("\n2. Testing Create Reminder")
        reminder_data = {
            "user_id": "test_patient_123",
            "title": "Take morning medication",
            "description": "Blood pressure medication (blue pill)",
            "scheduled_time": (datetime.now() + timedelta(hours=1)).isoformat(),
            "priority": "critical",
            "category": "medication",
            "caregiver_ids": ["caregiver_456"],
            "adaptive_scheduling_enabled": True,
            "escalation_enabled": True
        }
        
        response = requests.post(f"{BASE_URL}/api/reminders/create", json=reminder_data)
        if response.status_code in [200, 201]:
            print("✓ Reminder created successfully")
            print(f"   Response: {response.json()['message']}")
        else:
            print(f"✗ Failed to create reminder: {response.status_code}")
            print(f"   Error: {response.text}")
        
        # Test 3: Process user response
        print("\n3. Testing User Response Processing")
        response_data = {
            "reminder_id": "test_reminder_123",
            "user_id": "test_patient_123",
            "response_text": "Um... what medicine? I'm confused"
        }
        
        response = requests.post(f"{BASE_URL}/api/reminders/respond", json=response_data)
        if response.status_code == 200:
            result = response.json()
            print("✓ Response processed successfully")
            print(f"   Cognitive Risk: {result.get('cognitive_risk_score', 'N/A')}")
            print(f"   Interaction Type: {result.get('interaction_type', 'N/A')}")
            print(f"   Caregiver Notified: {result.get('caregiver_notified', False)}")
        else:
            print(f"✗ Failed to process response: {response.status_code}")
        
        # Test 4: Get behavior analytics
        print("\n4. Testing Behavior Analytics")
        response = requests.get(f"{BASE_URL}/api/reminders/behavior/test_patient_123?days=30")
        if response.status_code == 200:
            print("✓ Behavior analytics retrieved")
            pattern = response.json().get('pattern', {})
            print(f"   Total Reminders: {pattern.get('total_reminders', 0)}")
            print(f"   Confusion Trend: {pattern.get('confusion_trend', 'N/A')}")
        else:
            print(f"✗ Failed to get analytics: {response.status_code}")
        
        # Test 5: Natural language command
        print("\n5. Testing Natural Language Command")
        nl_data = {
            "user_id": "test_patient_123",
            "command_text": "Remind me to take my tablets after lunch"
        }
        
        response = requests.post(f"{BASE_URL}/api/reminders/natural-language", json=nl_data)
        if response.status_code == 200:
            print("✓ Natural language command processed")
            result = response.json()
            print(f"   Parsed: {result.get('parsed_reminder', {}).get('title', 'N/A')}")
        else:
            print(f"✗ Failed to process NL command: {response.status_code}")
        
        # Test 6: Dashboard analytics
        print("\n6. Testing Dashboard Analytics")
        response = requests.get(f"{BASE_URL}/api/reminders/analytics/dashboard/test_patient_123?days=7")
        if response.status_code == 200:
            print("✓ Dashboard analytics retrieved")
            dashboard = response.json().get('dashboard', {})
            stats = dashboard.get('statistics', {})
            print(f"   Total Reminders: {stats.get('total_reminders', 0)}")
            print(f"   Confirmation Rate: {stats.get('confirmation_rate', 0):.1%}")
        else:
            print(f"✗ Failed to get dashboard: {response.status_code}")
        
        print("\n" + "=" * 60)
        print("🎉 API Testing Complete!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("   Make sure the API is running: python -m uvicorn src.api.app_simple:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    test_api_endpoints()