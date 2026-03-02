"""
Behavioral Data Models (Step 1)

Defines all data structures for tracking:
- Wake / sleep times
- Medication reminder responses (responded or missed)
- Daily activity completion rates
- App interaction frequency
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActivityType(str, Enum):
    """Types of tracked daily activities."""
    WAKE_UP         = "wake_up"
    SLEEP           = "sleep"
    MEDICATION      = "medication"
    MEAL            = "meal"
    EXERCISE        = "exercise"
    HYGIENE         = "hygiene"
    APPOINTMENT     = "appointment"
    APP_INTERACTION = "app_interaction"
    REMINDER_RESPONSE = "reminder_response"
    GENERAL         = "general"


class ResponseType(str, Enum):
    """How a user responded to a reminder."""
    RESPONDED_ON_TIME  = "responded_on_time"
    RESPONDED_LATE     = "responded_late"
    MISSED             = "missed"
    DISMISSED          = "dismissed"
    CONFUSED           = "confused"


class DementiaRiskLevel(str, Enum):
    """Computed dementia risk level from behavioral deviation."""
    LOW      = "low"       # Deviation < 20% → routine is stable
    MEDIUM   = "medium"    # Deviation 20-50% → increase reminder frequency
    HIGH     = "high"      # Deviation > 50% → alert caregiver immediately


# ---------------------------------------------------------------------------
# Core log model (one event per entry)
# ---------------------------------------------------------------------------

class UserBehavioralLog(BaseModel):
    """
    A single behavioral event log entry for a user.

    One of these is created every time the user:
    - Wakes up / goes to sleep (from app interaction patterns)
    - Responds to (or misses) a medication reminder
    - Completes or skips a daily activity
    - Opens / closes the app
    """

    id: Optional[str] = None
    user_id: str = Field(..., description="Patient's user ID")

    # What happened
    activity_type: ActivityType = ActivityType.GENERAL
    response_type: Optional[ResponseType] = None  # Only for reminders

    # When it happened
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scheduled_time: Optional[datetime] = None  # Expected time (for deviation calc)

    # Deviation from scheduled time in minutes (positive = late, negative = early)
    time_deviation_minutes: Optional[float] = None

    # Activity completion (0.0 – 1.0)
    completion_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # Linked reminder (if any)
    reminder_id: Optional[str] = None
    reminder_category: Optional[str] = None

    # Extra context
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "patient_001",
                "activity_type": "medication",
                "response_type": "responded_late",
                "timestamp": "2026-03-02T08:42:00",
                "scheduled_time": "2026-03-02T08:00:00",
                "time_deviation_minutes": 42.0,
                "completion_rate": 1.0,
                "reminder_id": "rem_abc123",
                "reminder_category": "medication"
            }
        }


# ---------------------------------------------------------------------------
# Time-series aggregated view (input for Chronos)
# ---------------------------------------------------------------------------

class DailyBehaviorSummary(BaseModel):
    """
    Aggregated daily behavioral metrics for one user.
    This is what gets fed into the Chronos-T5-Small model.
    """
    date: str                            # "YYYY-MM-DD"
    user_id: str

    # Routine timing (hours from midnight)
    wake_time_hour: Optional[float] = None
    sleep_time_hour: Optional[float] = None

    # Medication
    medication_responses: int = 0        # Total reminders responded to
    medication_misses: int = 0           # Total reminders missed
    avg_medication_delay_minutes: float = 0.0

    # Activity
    activities_completed: int = 0
    activities_total: int = 0
    avg_completion_rate: float = 1.0

    # App engagement
    app_interactions: int = 0
    avg_response_delay_minutes: float = 0.0


class BehavioralTimeSeries(BaseModel):
    """
    A sequence of daily summaries for one user.
    Used as input to Chronos-T5-Small.
    """
    user_id: str
    days: List[DailyBehaviorSummary] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Chronos output
# ---------------------------------------------------------------------------

class BehaviorDeviationResult(BaseModel):
    """
    Result from Chronos-T5-Small: predicted vs actual behavior,
    and the computed deviation score.
    """
    user_id: str
    analysis_date: datetime = Field(default_factory=datetime.utcnow)

    # Raw data
    actual_values: List[float] = Field(default_factory=list)
    predicted_values: List[float] = Field(default_factory=list)

    # Deviation metrics
    mean_absolute_error: float = 0.0
    deviation_percentage: float = 0.0    # 0–100

    feature_analyzed: str = "avg_completion_rate"


# ---------------------------------------------------------------------------
# Final risk report
# ---------------------------------------------------------------------------

class DementiaRiskReport(BaseModel):
    """
    Final dementia risk report for a patient, combining
    behavioral deviation score with risk level and recommended action.
    """
    user_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Risk assessment
    risk_level: DementiaRiskLevel = DementiaRiskLevel.LOW
    deviation_percentage: float = 0.0

    # Breakdown per feature
    feature_scores: Dict[str, float] = Field(default_factory=dict)

    # Recommended actions
    recommended_action: str = ""
    increase_reminder_frequency: bool = False
    alert_caregiver: bool = False

    # Chatbot tone guidance (fed to susadi/hale-empathy-3b)
    empathy_tone: str = "supportive"   # "supportive" | "gentle_alert" | "urgent"

    # History
    previous_risk_level: Optional[DementiaRiskLevel] = None
    trend: str = "stable"   # "improving" | "stable" | "declining"
