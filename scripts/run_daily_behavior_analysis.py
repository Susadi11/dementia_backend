"""
Daily Behavioral Analysis Scheduler (Step 2 — Cron Job)

Runs the Chronos-T5-Small risk analysis for all active users
once per day. Call this script from Windows Task Scheduler or
any cron-like runner.

Usage (run manually):
    python scripts/run_daily_behavior_analysis.py

Usage (Windows Task Scheduler):
    Action:  "C:/Users/vindi perera/Documents/GitHub/dementia_backend/venv/Scripts/python.exe"
    Arguments: "C:/Users/vindi perera/Documents/GitHub/dementia_backend/scripts/run_daily_behavior_analysis.py"
    Trigger: Daily at 02:00 AM

Usage (Linux/Mac cron):
    0 2 * * * /path/to/venv/bin/python /path/to/scripts/run_daily_behavior_analysis.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).parent.parent / ".env"))

from src.database import Database
from src.services.behavioral_timeline_service import BehavioralTimelineService
from src.features.behavioral_analysis import DementiaRiskScorer

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_scheduler")


async def get_all_active_user_ids() -> list:
    """
    Fetch all distinct user IDs that have behavioral logs in the last 30 days.
    Only analyze users who have been actively logging data.
    """
    try:
        from datetime import timedelta
        col = Database.get_collection("behavioral_logs")
        since = datetime.utcnow() - timedelta(days=30)
        user_ids = await col.distinct("user_id", {"timestamp": {"$gte": since}})
        return user_ids
    except Exception as e:
        logger.error(f"Failed to fetch active users: {e}")
        return []


async def run_analysis_for_user(
    user_id: str,
    timeline_service: BehavioralTimelineService,
    risk_scorer: DementiaRiskScorer,
) -> None:
    """Run the full Chronos analysis pipeline for one user and save the report."""
    try:
        logger.info(f"Analyzing user: {user_id}")

        # Build behavioral time series from MongoDB (last 30 days)
        time_series = await timeline_service.get_time_series(user_id, days=30)

        if not time_series.days or len(time_series.days) < 7:
            logger.info(
                f"  Skipping {user_id} — only {len(time_series.days)} days of data "
                "(need at least 7)"
            )
            return

        # Get previous report for trend calculation
        previous_report = await timeline_service.get_latest_risk_report(user_id)

        # Run Chronos + risk scoring
        report = await risk_scorer.compute_risk(time_series, previous_report)

        # Persist report to MongoDB
        await timeline_service.save_risk_report(report)

        logger.info(
            f"  ✅ {user_id} → risk={report.risk_level.value} "
            f"deviation={report.deviation_percentage:.1f}% "
            f"trend={report.trend} "
            f"alert_caregiver={report.alert_caregiver}"
        )

        # If HIGH risk, trigger caregiver notification
        if report.alert_caregiver:
            try:
                from src.features.reminder_system.caregiver_notifier import CaregiverNotifier
                notifier = CaregiverNotifier()
                await notifier.send_risk_alert(
                    user_id=user_id,
                    risk_level=report.risk_level.value,
                    deviation=report.deviation_percentage,
                    message=report.recommended_action,
                )
                logger.warning(
                    f"  ⚠️  HIGH RISK caregiver alert sent for user={user_id}"
                )
            except Exception as notify_err:
                logger.error(f"  Caregiver notification failed: {notify_err}")

    except Exception as e:
        logger.error(f"  ❌ Failed to analyze {user_id}: {e}", exc_info=True)


async def main():
    logger.info("=" * 60)
    logger.info(f"Daily Behavioral Analysis — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    # Connect to MongoDB
    await Database.connect_to_database()

    timeline_service = BehavioralTimelineService()
    risk_scorer = DementiaRiskScorer()

    # Get all users with recent behavioral data
    user_ids = await get_all_active_user_ids()
    logger.info(f"Found {len(user_ids)} active user(s) to analyze")

    if not user_ids:
        logger.info("No users to analyze. Exiting.")
        return

    # Analyze each user sequentially (safe for CPU-bound Chronos model)
    success_count = 0
    for user_id in user_ids:
        await run_analysis_for_user(user_id, timeline_service, risk_scorer)
        success_count += 1

    await Database.close_database_connection()
    logger.info("=" * 60)
    logger.info(f"Analysis complete. Processed {success_count}/{len(user_ids)} users.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
