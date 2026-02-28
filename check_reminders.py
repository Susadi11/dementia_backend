"""Diagnostic: check reminders in DB for patient_001."""
import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from src.database import Database
from src.services.reminder_db_service import ReminderDatabaseService


async def check():
    await Database.connect_to_database()
    svc = ReminderDatabaseService()

    print(f"\nServer local time now : {datetime.now()}")

    # All reminders for patient_001
    reminders = await svc.get_user_reminders("patient_001", limit=20)
    print(f"\nTotal reminders for patient_001 (all statuses): {len(reminders)}")
    for r in reminders:
        print(f"  id={r['id']}  status={r.get('status')}  scheduled_time={r.get('scheduled_time')}")

    if not reminders:
        print("\n  *** NO REMINDERS FOUND — the 'reminders' collection is empty for patient_001 ***")
        print("  You need to create a reminder first via POST /api/reminders/create-from-text")
        return

    # Due in next 60 min with 60 min lookback (wide window to catch anything)
    due = await svc.get_due_reminders(
        time_window_minutes=60,
        user_id="patient_001",
        lookback_minutes=60,
    )
    print(f"\nDue within [-60 min, +60 min] window: {len(due)}")
    for r in due:
        print(f"  id={r['id']}  status={r.get('status')}  scheduled_time={r.get('scheduled_time')}")


asyncio.run(check())
