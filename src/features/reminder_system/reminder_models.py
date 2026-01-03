"""
Data models for the reminder system.

Defines structures for reminders, user interactions, and behavior tracking.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ReminderStatus(str, Enum):
    """Status of a reminder."""
    ACTIVE = "active"
    COMPLETED = "completed"
    MISSED = "missed"
    SNOOZED = "snoozed"
    CANCELLED = "cancelled"


class InteractionType(str, Enum):
    """Type of user interaction with reminder."""
    CONFIRMED = "confirmed"
    IGNORED = "ignored"
    DELAYED = "delayed"
    CONFUSED = "confused"
    REPEATED_QUESTION = "repeated_question"
    PARTIAL_COMPLETION = "partial_completion"


class ReminderPriority(str, Enum):
    """Priority level for reminders."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Reminder(BaseModel):
    """Reminder data model."""
    
    id: Optional[str] = None
    user_id: str
    title: str
    description: Optional[str] = None
    
    # Scheduling
    scheduled_time: datetime
    repeat_pattern: Optional[str] = None  # "daily", "weekly", "custom"
    repeat_interval_minutes: Optional[int] = None
    
    # Priority and categorization
    priority: ReminderPriority = ReminderPriority.MEDIUM
    category: str = "general"  # medication, appointment, meal, hygiene, etc.
    
    # Status
    status: ReminderStatus = ReminderStatus.ACTIVE
    
    # Adaptive features
    adaptive_scheduling_enabled: bool = True
    escalation_enabled: bool = True
    escalation_threshold_minutes: int = 30  # Time before caregiver notification
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Caregiver settings
    caregiver_ids: List[str] = Field(default_factory=list)
    notify_caregiver_on_miss: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "title": "Take morning medication",
                "description": "Blood pressure medication (blue pill)",
                "scheduled_time": "2025-11-25T08:00:00",
                "priority": "critical",
                "category": "medication",
                "caregiver_ids": ["caregiver456"]
            }
        }


class ReminderInteraction(BaseModel):
    """User interaction with a reminder."""
    
    id: Optional[str] = None
    reminder_id: str
    user_id: str
    reminder_category: Optional[str] = None
    
    # Interaction details
    interaction_type: InteractionType
    interaction_time: datetime = Field(default_factory=datetime.now)
    
    # User response
    user_response_text: Optional[str] = None
    user_response_audio_path: Optional[str] = None
    
    # Analysis results
    cognitive_risk_score: Optional[float] = None
    confusion_detected: bool = False
    memory_issue_detected: bool = False
    uncertainty_detected: bool = False
    
    # Features extracted
    features: Optional[Dict[str, float]] = None
    
    # Recommended action
    recommended_action: Optional[str] = None
    caregiver_alert_triggered: bool = False
    
    # Response time
    response_time_seconds: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "reminder_id": "rem123",
                "user_id": "user123",
                "interaction_type": "confused",
                "user_response_text": "Um... what medicine?",
                "cognitive_risk_score": 0.75,
                "confusion_detected": True,
                "recommended_action": "escalate_to_caregiver"
            }
        }


class BehaviorPattern(BaseModel):
    """User behavior pattern for a specific reminder or category."""
    
    user_id: str
    reminder_id: Optional[str] = None
    category: Optional[str] = None
    
    # Statistics
    total_reminders: int = 0
    confirmed_count: int = 0
    ignored_count: int = 0
    delayed_count: int = 0
    confused_count: int = 0
    
    # Timing patterns
    avg_response_time_seconds: Optional[float] = None
    optimal_reminder_hour: Optional[int] = None
    worst_response_hours: List[int] = Field(default_factory=list)
    
    # Cognitive indicators
    avg_cognitive_risk_score: Optional[float] = None
    confusion_trend: str = "stable"  # improving, stable, declining
    
    # Recommendations
    recommended_frequency_multiplier: float = 1.0
    recommended_time_adjustment_minutes: int = 0
    escalation_recommended: bool = False
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    analysis_period_days: int = 30


class CaregiverAlert(BaseModel):
    """Alert sent to caregiver."""
    
    id: Optional[str] = None
    caregiver_id: str
    user_id: str
    reminder_id: str
    
    # Alert details
    alert_type: str  # "missed_critical", "confusion_detected", "high_cognitive_risk"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    
    # Context
    reminder_title: str
    missed_count: int = 0
    cognitive_risk_score: Optional[float] = None
    user_response: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Status
    is_acknowledged: bool = False
    is_resolved: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "caregiver_id": "caregiver456",
                "user_id": "user123",
                "reminder_id": "rem123",
                "alert_type": "missed_critical",
                "severity": "high",
                "message": "Patient missed critical medication reminder 3 times",
                "reminder_title": "Take morning medication",
                "missed_count": 3
            }
        }


class ReminderCommand(BaseModel):
    """Natural language command for creating/modifying reminders."""
    
    user_id: str
    command_text: str
    audio_path: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "command_text": "Remind me to take my tablets after lunch"
            }
        }


class ReminderResponse(BaseModel):
    """User's response to a reminder notification."""
    
    reminder_id: str
    user_id: str
    response_text: Optional[str] = None
    audio_path: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "reminder_id": "rem123",
                "user_id": "user123",
                "response_text": "I already did it"
            }
        }
