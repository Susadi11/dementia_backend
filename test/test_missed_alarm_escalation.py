"""
Test Missed Alarm Escalation

Simulates a reminder being missed and verifies:
1. POST /api/reminders/missed/{reminder_id} returns caregiver_notified=True
2. A caregiver_alerts document is inserted in MongoDB

Requires:
- The API server running on localhost:8080
- MongoDB accessible (MONGODB_URI in .env)
"""

import os
import sys
import requests
import json
from datetime import datetime, timedelta
from uuid import uuid4

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pymongo import MongoClient

BASE_URL = "http://localhost:8080"
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "dementia_care_db")


def print_test(name, passed, detail=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} - {name}")
    if detail:
        print(f"        {detail}")


def run_tests():
    print("=" * 60)
    print("  Missed Alarm Escalation Test")
    print("=" * 60)

    # ── unique ids for this run ──────────────────────────────
    run_tag = uuid4().hex[:8]
    reminder_id = f"test_missed_{run_tag}"
    patient_id = f"test_patient_{run_tag}"
    caregiver_id = f"test_caregiver_{run_tag}"

    results = []

    # ── 1. Create a reminder with caregiver_ids ──────────────
    print("\n[Step 1] Create reminder")
    reminder_data = {
        "id": reminder_id,
        "user_id": patient_id,
        "title": "Escalation Test Medication",
        "description": "Test reminder for missed-alarm escalation",
        "scheduled_time": (datetime.now() + timedelta(hours=1)).isoformat(),
        "priority": "high",
        "category": "medication",
        "recurrence": "none",
        "caregiver_ids": [caregiver_id],
        "notify_caregiver_on_miss": True,
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/reminders/create", json=reminder_data)
        created = resp.status_code == 201
        print_test("Create reminder", created, f"status={resp.status_code}")
        results.append(created)
    except Exception as e:
        print_test("Create reminder", False, str(e))
        results.append(False)

    # ── 2. Mark reminder missed (simulates 3 exhausted attempts) ─
    print("\n[Step 2] Mark reminder as missed")
    caregiver_notified = False
    try:
        resp = requests.post(
            f"{BASE_URL}/api/reminders/missed/{reminder_id}",
            params={"user_id": patient_id},
        )
        ok = resp.status_code == 200
        body = resp.json() if ok else {}
        caregiver_notified = body.get("caregiver_notified", False)

        print_test("Missed endpoint returns 200", ok, f"status={resp.status_code}")
        results.append(ok)

        print_test(
            "caregiver_notified is True",
            caregiver_notified,
            f"caregiver_notified={caregiver_notified}",
        )
        results.append(caregiver_notified)

        attempts = body.get("attempts", 0)
        print_test(
            "attempts >= 3",
            attempts >= 3,
            f"attempts={attempts}",
        )
        results.append(attempts >= 3)

        if ok:
            print(f"        Response: {json.dumps(body, indent=2, default=str)}")
    except Exception as e:
        print_test("Missed endpoint", False, str(e))
        results.extend([False, False, False])

    # ── 3. Verify caregiver_alerts in MongoDB ────────────────
    print("\n[Step 3] Verify caregiver_alerts in MongoDB")
    alert_found = False
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        alerts_col = db["caregiver_alerts"]

        alert_doc = alerts_col.find_one({
            "reminder_id": reminder_id,
            "patient_id": patient_id,
            "caregiver_id": caregiver_id,
        })
        alert_found = alert_doc is not None
        print_test("Alert document exists in DB", alert_found)

        if alert_found:
            print_test(
                "alert_type is missed_alarm",
                alert_doc.get("alert_type") == "missed_alarm",
                f"alert_type={alert_doc.get('alert_type')}",
            )
            results.append(alert_doc.get("alert_type") == "missed_alarm")

            print_test(
                "resolved is False",
                alert_doc.get("resolved") is False,
                f"resolved={alert_doc.get('resolved')}",
            )
            results.append(alert_doc.get("resolved") is False)
        else:
            results.extend([False, False])

        results.append(alert_found)

        # ── Cleanup test data ────────────────────────────────
        alerts_col.delete_many({"reminder_id": reminder_id})
        db["reminders"].delete_many({"id": reminder_id})
        db["reminder_interactions"].delete_many({"reminder_id": reminder_id})
        client.close()
        print("\n  [Cleanup] Test documents removed from DB")
    except Exception as e:
        print_test("MongoDB verification", False, str(e))
        results.extend([False, False, False])

    # ── Summary ──────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed")
    print("=" * 60)
    return all(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
