"""
Behavioral Analysis API Routes (Step 5)

Endpoints:
  POST /behavior/log                   – Log a single behavioral event
  POST /behavior/analyze/{user_id}     – Run Chronos + risk scoring
  GET  /behavior/risk/{user_id}        – Get latest risk report
  GET  /behavior/risk/history/{user_id}– Get risk history (trend)
  GET  /behavior/series/{user_id}      – Get raw behavioral time series

These endpoints wire together:
  - BehavioralTimelineService  (MongoDB)
  - ChronosAnalyzer             (amazon/chronos-t5-small)
  - DementiaRiskScorer          (risk level + actions)
  - Existing caregiver notifier (CaregiverNotifier from reminder_system)
  - susadi/hale-empathy-3b      (empathy tone is returned for chatbot)
"""

import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel

from src.features.behavioral_analysis import (
    UserBehavioralLog,
    BehavioralTimeSeries,
    DementiaRiskReport,
    DementiaRiskLevel,
    ChronosAnalyzer,
    DementiaRiskScorer,
)
from src.features.behavioral_analysis.behavioral_models import (
    ActivityType,
    ResponseType,
    DailyBehaviorSummary,
)
from src.services.behavioral_timeline_service import BehavioralTimelineService
from src.features.reminder_system.caregiver_notifier import CaregiverNotifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/behavior", tags=["Behavioral Analysis"])

# Shared service instances
_timeline_service: Optional[BehavioralTimelineService] = None
_risk_scorer: Optional[DementiaRiskScorer] = None
_caregiver_notifier: Optional[CaregiverNotifier] = None


def _get_services():
    global _timeline_service, _risk_scorer, _caregiver_notifier
    if _timeline_service is None:
        _timeline_service = BehavioralTimelineService()
    if _risk_scorer is None:
        _risk_scorer = DementiaRiskScorer()
    if _caregiver_notifier is None:
        _caregiver_notifier = CaregiverNotifier()
    return _timeline_service, _risk_scorer, _caregiver_notifier


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class LogEventRequest(BaseModel):
    """Request body for logging a single behavioral event."""
    user_id: str
    activity_type: ActivityType = ActivityType.GENERAL
    response_type: Optional[ResponseType] = None
    timestamp: Optional[datetime] = None
    scheduled_time: Optional[datetime] = None
    time_deviation_minutes: Optional[float] = None
    completion_rate: float = 1.0
    reminder_id: Optional[str] = None
    reminder_category: Optional[str] = None
    notes: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response from behavioral analysis endpoint."""
    user_id: str
    risk_level: DementiaRiskLevel
    deviation_percentage: float
    feature_scores: dict
    recommended_action: str
    increase_reminder_frequency: bool
    alert_caregiver: bool
    empathy_tone: str
    trend: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# POST /behavior/log
# ---------------------------------------------------------------------------

@router.post("/log", summary="Log a behavioral event")
async def log_behavioral_event(request: LogEventRequest):
    """
    Log one behavioral event for a user.

    Call this whenever:
    - User responds to a reminder (with response_type = responded_on_time / responded_late / missed)
    - User opens the app (activity_type = app_interaction)
    - User completes a daily activity (with completion_rate)
    """
    try:
        timeline_service, _, _ = _get_services()

        log = UserBehavioralLog(
            user_id=request.user_id,
            activity_type=request.activity_type,
            response_type=request.response_type,
            timestamp=request.timestamp or datetime.utcnow(),
            scheduled_time=request.scheduled_time,
            time_deviation_minutes=request.time_deviation_minutes,
            completion_rate=request.completion_rate,
            reminder_id=request.reminder_id,
            reminder_category=request.reminder_category,
            notes=request.notes,
        )

        event_id = await timeline_service.log_event(log)
        return {"status": "logged", "event_id": event_id, "user_id": request.user_id}

    except Exception as e:
        logger.error(f"Error logging behavioral event: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log event: {str(e)}"
        )


# ---------------------------------------------------------------------------
# POST /behavior/analyze/{user_id}
# ---------------------------------------------------------------------------

@router.post(
    "/analyze/{user_id}",
    response_model=AnalysisResponse,
    summary="Run Chronos analysis and compute dementia risk"
)
async def analyze_behavior(
    user_id: str,
    days: int = Query(default=30, ge=7, le=90, description="Days of history to analyze")
):
    """
    Full pipeline for one user:
      1. Fetch behavioral time series from MongoDB (last N days)
      2. Feed to amazon/chronos-t5-small → get predicted behavior
      3. Compute deviation (actual vs predicted)
      4. Score risk level: LOW / MEDIUM / HIGH
      5. If HIGH → automatically notify caregiver
      6. Return report with empathy_tone for chatbot

    The empathy_tone field should be passed to susadi/hale-empathy-3b
    as a system prompt hint to adjust the chatbot's conversation style.
    """
    try:
        timeline_service, risk_scorer, caregiver_notifier = _get_services()

        # Get behavioral time series from MongoDB
        time_series: BehavioralTimeSeries = await timeline_service.get_time_series(
            user_id, days=days
        )

        if not time_series.days:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No behavioral data found for user {user_id}. "
                       "Start logging events via POST /behavior/log"
            )

        # Get previous risk report for trend calculation
        previous_report = await timeline_service.get_latest_risk_report(user_id)

        # Run Chronos + risk scoring
        report: DementiaRiskReport = await risk_scorer.compute_risk(
            time_series, previous_report
        )

        # Persist the new risk report
        await timeline_service.save_risk_report(report)

        # === Step 5: Connect to existing caregiver notifier ===
        if report.alert_caregiver:
            logger.warning(
                f"HIGH RISK detected for user={user_id}. "
                f"Sending caregiver alert."
            )
            try:
                await caregiver_notifier.send_risk_alert(
                    user_id=user_id,
                    risk_level=report.risk_level.value,
                    deviation=report.deviation_percentage,
                    message=report.recommended_action,
                )
            except Exception as notify_err:
                logger.error(f"Failed to notify caregiver: {notify_err}")

        return AnalysisResponse(
            user_id=report.user_id,
            risk_level=report.risk_level,
            deviation_percentage=report.deviation_percentage,
            feature_scores=report.feature_scores,
            recommended_action=report.recommended_action,
            increase_reminder_frequency=report.increase_reminder_frequency,
            alert_caregiver=report.alert_caregiver,
            empathy_tone=report.empathy_tone,
            trend=report.trend,
            generated_at=report.generated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing behavior for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# GET /behavior/risk/{user_id}
# ---------------------------------------------------------------------------

@router.get(
    "/risk/{user_id}",
    response_model=AnalysisResponse,
    summary="Get the latest dementia risk report for a user"
)
async def get_latest_risk(user_id: str):
    """Return the most recent risk report without re-running analysis."""
    try:
        timeline_service, _, _ = _get_services()
        report = await timeline_service.get_latest_risk_report(user_id)

        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No risk report found for user {user_id}. "
                       "Run POST /behavior/analyze/{user_id} first."
            )

        return AnalysisResponse(
            user_id=report.user_id,
            risk_level=report.risk_level,
            deviation_percentage=report.deviation_percentage,
            feature_scores=report.feature_scores,
            recommended_action=report.recommended_action,
            increase_reminder_frequency=report.increase_reminder_frequency,
            alert_caregiver=report.alert_caregiver,
            empathy_tone=report.empathy_tone,
            trend=report.trend,
            generated_at=report.generated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching risk report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch risk report"
        )


# ---------------------------------------------------------------------------
# GET /behavior/risk/history/{user_id}
# ---------------------------------------------------------------------------

@router.get(
    "/risk/history/{user_id}",
    summary="Get dementia risk history for trend visualization"
)
async def get_risk_history(
    user_id: str,
    limit: int = Query(default=30, le=90)
):
    """
    Returns the last N risk reports for dashboard trend charts.
    Shows if the patient is improving, stable, or declining over time.
    """
    try:
        timeline_service, _, _ = _get_services()
        history = await timeline_service.get_risk_history(user_id, limit=limit)

        return {
            "user_id": user_id,
            "total_reports": len(history),
            "history": [
                {
                    "date": r.generated_at.strftime("%Y-%m-%d"),
                    "risk_level": r.risk_level.value,
                    "deviation_percentage": r.deviation_percentage,
                    "trend": r.trend,
                }
                for r in history
            ]
        }

    except Exception as e:
        logger.error(f"Error fetching risk history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch risk history"
        )


# ---------------------------------------------------------------------------
# GET /behavior/series/{user_id}
# ---------------------------------------------------------------------------

@router.get(
    "/series/{user_id}",
    summary="Get behavioral time series data for a user"
)
async def get_behavioral_series(
    user_id: str,
    days: int = Query(default=30, ge=1, le=90)
):
    """
    Returns the raw aggregated daily behavioral summaries.
    Useful for debugging or building custom dashboards.
    """
    try:
        timeline_service, _, _ = _get_services()
        series = await timeline_service.get_time_series(user_id, days=days)

        return {
            "user_id": user_id,
            "days_of_data": len(series.days),
            "series": [d.dict() for d in series.days],
        }

    except Exception as e:
        logger.error(f"Error fetching behavioral series: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch behavioral series"
        )
