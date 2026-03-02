"""
Behavioral Timeline Database Service (Step 2)

Stores and retrieves user behavioral logs and daily summaries
from MongoDB.

Collections used:
  - behavioral_logs          : raw per-event logs (UserBehavioralLog)
  - behavioral_daily_summary : aggregated daily summaries (DailyBehaviorSummary)
  - dementia_risk_reports    : computed risk reports (DementiaRiskReport)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from bson import ObjectId

from src.database import Database
from src.features.behavioral_analysis.behavioral_models import (
    UserBehavioralLog,
    DailyBehaviorSummary,
    BehavioralTimeSeries,
    DementiaRiskReport,
    ActivityType,
    ResponseType,
)

logger = logging.getLogger(__name__)


class BehavioralTimelineService:
    """
    All MongoDB operations for the behavioral analysis pipeline.
    
    Usage:
        service = BehavioralTimelineService()
        await service.log_event(log_entry)
        series  = await service.get_time_series(user_id, days=30)
    """

    def __init__(self):
        self._logs_col = None
        self._daily_col = None
        self._risk_col = None

    # ------------------------------------------------------------------
    # Lazy-loaded collections
    # ------------------------------------------------------------------

    @property
    def logs_collection(self):
        if self._logs_col is None:
            self._logs_col = Database.get_collection("behavioral_logs")
        return self._logs_col

    @property
    def daily_collection(self):
        if self._daily_col is None:
            self._daily_col = Database.get_collection("behavioral_daily_summary")
        return self._daily_col

    @property
    def risk_collection(self):
        if self._risk_col is None:
            self._risk_col = Database.get_collection("dementia_risk_reports")
        return self._risk_col

    # ------------------------------------------------------------------
    # Step 2a: Log raw behavioral event
    # ------------------------------------------------------------------

    async def log_event(self, log: UserBehavioralLog) -> str:
        """
        Save a single behavioral event to MongoDB.
        Called every time user interacts with a reminder or completes an activity.

        Returns: inserted document id
        """
        try:
            doc = log.dict()
            if not doc.get("id"):
                doc["id"] = str(ObjectId())
            doc["_id"] = doc["id"]

            await self.logs_collection.insert_one(doc)
            logger.info(f"Logged behavioral event for user={log.user_id} type={log.activity_type}")
            return doc["id"]

        except Exception as e:
            logger.error(f"Failed to log behavioral event: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Step 2b: Get raw logs for a user within N days
    # ------------------------------------------------------------------

    async def get_raw_logs(
        self,
        user_id: str,
        days: int = 30
    ) -> List[UserBehavioralLog]:
        """Retrieve all raw behavioral logs for user in last N days."""
        try:
            since = datetime.utcnow() - timedelta(days=days)
            cursor = self.logs_collection.find({
                "user_id": user_id,
                "timestamp": {"$gte": since}
            }).sort("timestamp", 1)

            docs = await cursor.to_list(length=1000)
            return [UserBehavioralLog(**d) for d in docs]

        except Exception as e:
            logger.error(f"Failed to fetch raw logs: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Step 2c: Build daily aggregated summaries from raw logs
    # ------------------------------------------------------------------

    async def build_daily_summaries(
        self,
        user_id: str,
        days: int = 30
    ) -> List[DailyBehaviorSummary]:
        """
        Aggregate raw logs into daily summaries that Chronos can consume.
        Groups events by calendar day and computes:
          - Wake/sleep hours
          - Medication response counts and delays
          - Activity completion rate
          - App interaction count
        """
        logs = await self.get_raw_logs(user_id, days)
        if not logs:
            return []

        # Group by date string
        day_map: Dict[str, List[UserBehavioralLog]] = {}
        for log in logs:
            date_str = log.timestamp.strftime("%Y-%m-%d")
            day_map.setdefault(date_str, []).append(log)

        summaries: List[DailyBehaviorSummary] = []

        for date_str, day_logs in sorted(day_map.items()):
            wake_times = [
                l.timestamp.hour + l.timestamp.minute / 60
                for l in day_logs if l.activity_type == ActivityType.WAKE_UP
            ]
            sleep_times = [
                l.timestamp.hour + l.timestamp.minute / 60
                for l in day_logs if l.activity_type == ActivityType.SLEEP
            ]
            med_logs = [l for l in day_logs if l.activity_type == ActivityType.MEDICATION]
            med_responded = [
                l for l in med_logs
                if l.response_type in (
                    ResponseType.RESPONDED_ON_TIME,
                    ResponseType.RESPONDED_LATE
                )
            ]
            med_missed = [
                l for l in med_logs
                if l.response_type == ResponseType.MISSED
            ]
            med_delays = [
                l.time_deviation_minutes for l in med_responded
                if l.time_deviation_minutes is not None
            ]

            activity_logs = [
                l for l in day_logs
                if l.activity_type not in (
                    ActivityType.APP_INTERACTION,
                    ActivityType.WAKE_UP,
                    ActivityType.SLEEP
                )
            ]
            completion_rates = [l.completion_rate for l in activity_logs]

            app_logs = [
                l for l in day_logs
                if l.activity_type == ActivityType.APP_INTERACTION
            ]
            response_delays = [
                l.time_deviation_minutes for l in day_logs
                if l.time_deviation_minutes is not None
            ]

            summaries.append(DailyBehaviorSummary(
                date=date_str,
                user_id=user_id,
                wake_time_hour=sum(wake_times) / len(wake_times) if wake_times else None,
                sleep_time_hour=sum(sleep_times) / len(sleep_times) if sleep_times else None,
                medication_responses=len(med_responded),
                medication_misses=len(med_missed),
                avg_medication_delay_minutes=sum(med_delays) / len(med_delays) if med_delays else 0.0,
                activities_completed=len([l for l in activity_logs if l.completion_rate >= 1.0]),
                activities_total=len(activity_logs),
                avg_completion_rate=sum(completion_rates) / len(completion_rates) if completion_rates else 1.0,
                app_interactions=len(app_logs),
                avg_response_delay_minutes=sum(response_delays) / len(response_delays) if response_delays else 0.0,
            ))

        return summaries

    # ------------------------------------------------------------------
    # Step 2d: Build BehavioralTimeSeries (Chronos input object)
    # ------------------------------------------------------------------

    async def get_time_series(
        self,
        user_id: str,
        days: int = 30
    ) -> BehavioralTimeSeries:
        """Return a BehavioralTimeSeries ready to pass to ChronosAnalyzer."""
        summaries = await self.build_daily_summaries(user_id, days)
        return BehavioralTimeSeries(user_id=user_id, days=summaries)

    # ------------------------------------------------------------------
    # Step 4 support: Save / retrieve risk reports
    # ------------------------------------------------------------------

    async def save_risk_report(self, report: DementiaRiskReport) -> str:
        """Persist a DementiaRiskReport to MongoDB."""
        try:
            doc = report.dict()
            doc["_id"] = f"{report.user_id}_{report.generated_at.strftime('%Y%m%d%H%M%S')}"

            await self.risk_collection.insert_one(doc)
            logger.info(f"Saved risk report for user={report.user_id} level={report.risk_level}")
            return doc["_id"]

        except Exception as e:
            logger.error(f"Failed to save risk report: {e}", exc_info=True)
            raise

    async def get_latest_risk_report(
        self,
        user_id: str
    ) -> Optional[DementiaRiskReport]:
        """Get the most recent risk report for a user."""
        try:
            doc = await self.risk_collection.find_one(
                {"user_id": user_id},
                sort=[("generated_at", -1)]
            )
            return DementiaRiskReport(**doc) if doc else None

        except Exception as e:
            logger.error(f"Failed to fetch risk report: {e}", exc_info=True)
            return None

    async def get_risk_history(
        self,
        user_id: str,
        limit: int = 30
    ) -> List[DementiaRiskReport]:
        """Get risk history for trend analysis."""
        try:
            cursor = self.risk_collection.find(
                {"user_id": user_id}
            ).sort("generated_at", -1).limit(limit)

            docs = await cursor.to_list(length=limit)
            return [DementiaRiskReport(**d) for d in docs]

        except Exception as e:
            logger.error(f"Failed to fetch risk history: {e}", exc_info=True)
            return []
