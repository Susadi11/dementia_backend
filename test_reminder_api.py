"""
Test script for the Reminder API

Tests basic CRUD operations for the reminder system.
Run this after starting the API server.
"""

import requests
import json
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8000/api/reminders"

def test_reminder_api():
    """Test the reminder API endpoints."""
    
    print("=== Testing Reminder API ===\n")
    
    # Test data
    test_user_id = "test_user_123"
    reminder_data = {
        "user_id": test_user_id,
        "title": "Take morning medication",
        "description": "Blood pressure medication - 2 blue pills",
        "scheduled_time": (datetime.now() + timedelta(hours=1)).isoformat(),
        "priority": "high",
        "category": "medication",
        "caregiver_ids": ["caregiver_456"],
        "notify_caregiver_on_miss": True,
        "escalation_threshold_minutes": 30
    }
    
    # 1. Test Health Check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 2. Test Create Reminder
    print("2. Testing create reminder...")
    try:
        response = requests.post(f"{BASE_URL}/", json=reminder_data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 201 and result.get("status") == "success":
            reminder_id = result["data"]["id"]
            print(f"   Created reminder with ID: {reminder_id}\n")
        else:
            print("   Failed to create reminder\n")
            return
            
    except Exception as e:
        print(f"   Error: {e}\n")
        return
    
    # 3. Test Get Reminder by ID
    print("3. Testing get reminder by ID...")
    try:
        response = requests.get(f"{BASE_URL}/{reminder_id}")
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 4. Test Get User Reminders
    print("4. Testing get user reminders...")
    try:
        response = requests.get(f"{BASE_URL}/user/{test_user_id}")
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 5. Test Update Reminder
    print("5. Testing update reminder...")
    try:
        update_data = reminder_data.copy()
        update_data["title"] = "Take evening medication"
        update_data["priority"] = "critical"
        
        response = requests.put(f"{BASE_URL}/{reminder_id}", json=update_data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 6. Test Complete Reminder
    print("6. Testing complete reminder...")
    try:
        response = requests.post(f"{BASE_URL}/{reminder_id}/complete")
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 7. Test Get Due Reminders
    print("7. Testing get due reminders...")
    try:
        response = requests.get(f"{BASE_URL}/due/now?time_window_minutes=60")
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 8. Test User Statistics
    print("8. Testing user statistics...")
    try:
        response = requests.get(f"{BASE_URL}/stats/user/{test_user_id}")
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 9. Test Record Interaction
    print("9. Testing record interaction...")
    try:
        interaction_data = {
            "reminder_id": reminder_id,
            "user_id": test_user_id,
            "interaction_type": "confirmed",
            "user_response_text": "Yes, I took my medication"
        }
        
        response = requests.post(f"{BASE_URL}/interactions", json=interaction_data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Response: {json.dumps(result, indent=2)}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 10. Test Delete Reminder (optional - uncomment to test)
    # print("10. Testing delete reminder...")
    # try:
    #     response = requests.delete(f"{BASE_URL}/{reminder_id}")
    #     print(f"   Status: {response.status_code}")
    #     result = response.json()
    #     print(f"   Response: {json.dumps(result, indent=2)}\n")
    # except Exception as e:
    #     print(f"   Error: {e}\n")
    
    print("=== Testing Complete ===")


if __name__ == "__main__":
    test_reminder_api()