"""
Reminder System API Routes

FastAPI endpoints for the Context-Aware Smart Reminder System.

Provides endpoints for:
- Creating, updating, and managing reminders
- Processing user responses
- Getting behavior analytics
- Caregiver notifications
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from typing import List, Optional
from datetime import datetime
import logging
import uuid
import tempfile
import os
from pathlib import Path

from src.features.reminder_system import (
    Reminder, ReminderCommand, ReminderResponse, ReminderInteraction,
    PittBasedReminderAnalyzer, AdaptiveReminderScheduler, BehaviorTracker
)
from src.features.reminder_system.caregiver_notifier import CaregiverNotifier
from src.features.reminder_system.reminder_models import (
    ReminderStatus, ReminderPriority, CaregiverAlert
)
from src.features.reminder_system.weekly_report_generator import (
    WeeklyReportGenerator, WeeklyCognitiveReport
)
from src.services.reminder_db_service import ReminderDatabaseService
from src.services.whisper_service import get_whisper_service
from src.features.conversational_ai.nlp.nlp_engine import NLPEngine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/reminders", tags=["reminders"])

# Initialize components (these would typically come from dependency injection)
reminder_analyzer = PittBasedReminderAnalyzer()
behavior_tracker = BehaviorTracker()
scheduler = AdaptiveReminderScheduler(behavior_tracker, reminder_analyzer)
caregiver_notifier = CaregiverNotifier()
report_generator = WeeklyReportGenerator(behavior_tracker)
nlp_engine = NLPEngine()

# Database service - initialized lazily
_db_service = None

def _parse_reminder_from_text(
    text: str,
    user_id: str,
    priority_override: Optional[str] = None
) -> dict:
    """
    Parse reminder details from natural language text using NLP.
    
    Extracts:
    - Title and description
    - Category (medication, appointment, meal, etc.)
    - Time/date information
    - Recurrence pattern
    - Priority level
    """
    from datetime import timedelta
    import re
    
    text_lower = text.lower()
    
    # Extract category based on keywords
    category = "general"
    if any(word in text_lower for word in ["medicine", "medication", "pill", "tablet", "drug"]):
        category = "medication"
    elif any(word in text_lower for word in ["doctor", "appointment", "visit", "checkup"]):
        category = "appointment"
    elif any(word in text_lower for word in ["breakfast", "lunch", "dinner", "meal", "eat"]):
        category = "meal"
    elif any(word in text_lower for word in ["shower", "bath", "hygiene", "brush", "wash"]):
        category = "hygiene"
    elif any(word in text_lower for word in ["exercise", "walk", "activity"]):
        category = "activity"
    
    # Extract priority
    priority = priority_override or "medium"
    if any(word in text_lower for word in ["urgent", "critical", "important", "asap"]):
        priority = "high"
    elif any(word in text_lower for word in ["when you can", "sometime", "eventually"]):
        priority = "low"
    
    # Extract recurrence pattern
    recurrence = None
    if any(word in text_lower for word in ["every day", "daily", "everyday"]):
        recurrence = "daily"
    elif any(word in text_lower for word in ["every week", "weekly"]):
        recurrence = "weekly"
    elif any(word in text_lower for word in ["every month", "monthly"]):
        recurrence = "monthly"
    
    # Extract time information (basic parsing)
    scheduled_time = datetime.now() + timedelta(hours=1)  # Default: 1 hour from now
    
    # Look for time patterns like "8 AM", "2:30 PM", "at noon"
    time_patterns = [
        r'(\d{1,2})\s*(?::|\.)\s*(\d{2})\s*(am|pm)',  # 2:30 PM
        r'(\d{1,2})\s*(am|pm)',  # 8 AM
        r'at\s+(\d{1,2})(?:\s*o\'?clock)?',  # at 8 o'clock
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                if len(match.groups()) == 3:  # HH:MM AM/PM
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    period = match.group(3)
                    if period == 'pm' and hour < 12:
                        hour += 12
                    elif period == 'am' and hour == 12:
                        hour = 0
                elif len(match.groups()) == 2:  # HH AM/PM
                    hour = int(match.group(1))
                    minute = 0
                    period = match.group(2)
                    if period == 'pm' and hour < 12:
                        hour += 12
                    elif period == 'am' and hour == 12:
                        hour = 0
                else:  # Just hour
                    hour = int(match.group(1))
                    minute = 0
                
                # Set the time for today or tomorrow if time has passed
                scheduled_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                if scheduled_time < datetime.now():
                    scheduled_time += timedelta(days=1)
                break
            except (ValueError, AttributeError):
                pass
    
    # Special time keywords
    if "noon" in text_lower or "lunch time" in text_lower:
        scheduled_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        if scheduled_time < datetime.now():
            scheduled_time += timedelta(days=1)
    elif "morning" in text_lower:
        scheduled_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        if scheduled_time < datetime.now():
            scheduled_time += timedelta(days=1)
    elif "evening" in text_lower or "dinner" in text_lower:
        scheduled_time = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
        if scheduled_time < datetime.now():
            scheduled_time += timedelta(days=1)
    elif "night" in text_lower or "bedtime" in text_lower:
        scheduled_time = datetime.now().replace(hour=21, minute=0, second=0, microsecond=0)
        if scheduled_time < datetime.now():
            scheduled_time += timedelta(days=1)
    
    # Generate title (extract main action)
    title = text[:50]  # First 50 chars
    if category == "medication":
        title = "Take Medication"
    elif category == "appointment":
        title = "Appointment Reminder"
    elif category == "meal":
        title = "Meal Reminder"
    
    # Add time context to title if recurrence exists
    if recurrence:
        title = f"{title} ({recurrence.capitalize()})"
    
    return {
        "title": title,
        "description": text,
        "category": category,
        "priority": priority,
        "scheduled_time": scheduled_time,
        "recurrence": recurrence
    }


def get_db_service():
    """Get database service instance (lazy initialization)."""
    global _db_service
    if _db_service is None:
        _db_service = ReminderDatabaseService()
    return _db_service


@router.post("/create", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_reminder(reminder: Reminder):
    """
    Create a new reminder for a user.
    
    - **user_id**: User identifier
    - **title**: Reminder title
    - **description**: Optional detailed description
    - **scheduled_time**: When to send reminder
    - **priority**: Priority level (low, medium, high, critical)
    - **category**: Reminder category (medication, appointment, meal, etc.)
    - **caregiver_ids**: List of caregiver IDs to notify
    """
    try:
        # Auto-generate ID if not provided
        if not reminder.id:
            reminder.id = f"reminder_{uuid.uuid4().hex[:12]}"
        
        # Set creation timestamps
        reminder.created_at = datetime.now()
        reminder.updated_at = datetime.now()
        
        # Save to MongoDB database
        db_service = get_db_service()
        db_result = await db_service.create_reminder(reminder)
        
        logger.info(f"Created reminder {reminder.id} for user {reminder.user_id}")
        
        return {
            "status": "success",
            "message": "Reminder created successfully",
            "reminder": reminder.dict(),
            "database_id": db_result.get("id")
        }
        
    except Exception as e:
        logger.error(f"Error creating reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/create-from-audio", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_reminder_from_audio(
    user_id: str = Form(..., description="User identifier"),
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, flac)"),
    priority: Optional[str] = Form("medium", description="Reminder priority (low, medium, high, critical)"),
    caregiver_ids: Optional[str] = Form(None, description="Comma-separated caregiver IDs")
):
    """
    Create reminder from audio recording using Whisper transcription + NLP parsing.
    
    **Flow:**
    1. Upload audio file
    2. Transcribe using Whisper (local, free)
    3. Parse reminder details from transcription (NLP)
    4. Create reminder automatically
    
    **Example audio commands:**
    - "Remind me to take my blood pressure medicine at 8 AM every morning"
    - "Set a reminder for doctor appointment next Tuesday at 2 PM"
    - "I need to remember my lunch at noon daily"
    
    **Parameters:**
    - **user_id**: User identifier
    - **file**: Audio file containing the reminder command
    - **priority**: Optional priority override (default: extracted from audio)
    - **caregiver_ids**: Optional caregiver IDs to notify (comma-separated)
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        logger.info(f"Creating reminder from audio for user: {user_id}")
        
        # Get file extension
        file_ext = Path(file.filename).suffix or ".wav"
        
        # Log file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        logger.info(f"Processing audio file... Size: {file_size} bytes")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        try:
            # Step 1: Transcribe audio using Whisper
            logger.info("📝 Transcribing audio with Whisper...")
            whisper_service = get_whisper_service()
            transcription_result = whisper_service.transcribe(
                audio_path=temp_audio_path,
                language=None  # Auto-detect
            )
            
            transcribed_text = transcription_result["text"]
            logger.info(f"✅ Transcription: '{transcribed_text}'")
            
            if not transcribed_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Could not transcribe audio. Please ensure clear speech."
                )
            
            # Step 2: Parse reminder details using NLP
            logger.info("🧠 Parsing reminder details from transcription...")
            reminder_details = _parse_reminder_from_text(
                text=transcribed_text,
                user_id=user_id,
                priority_override=priority
            )
            
            # Step 3: Create reminder
            reminder_id = f"reminder_{uuid.uuid4().hex[:12]}"
            
            # Parse caregiver IDs if provided
            caregiver_list = []
            if caregiver_ids:
                caregiver_list = [cid.strip() for cid in caregiver_ids.split(",") if cid.strip()]
            
            reminder = Reminder(
                id=reminder_id,
                user_id=user_id,
                title=reminder_details["title"],
                description=reminder_details["description"],
                scheduled_time=reminder_details["scheduled_time"],
                priority=ReminderPriority(reminder_details["priority"]),
                category=reminder_details["category"],
                recurrence=reminder_details.get("recurrence"),
                caregiver_ids=caregiver_list,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Save to database
            db_service = get_db_service()
            db_result = await db_service.create_reminder(reminder)
            
            logger.info(f"✅ Reminder created from audio: {reminder_id}")
            
            return {
                "status": "success",
                "message": "Reminder created successfully from audio",
                "reminder": reminder.dict(),
                "transcription": transcribed_text,
                "audio_file": file.filename,
                "db_status": db_result
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating reminder from audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process audio: {str(e)}"
        )


@router.post("/natural-language", response_model=dict)
async def create_reminder_from_natural_language(command: ReminderCommand):
    """
    Create reminder from natural language text command using NLP.
    
    Examples:
    - "Remind me to take my tablets after lunch"
    - "Set medication reminder for 8 AM daily"
    - "Doctor appointment next Tuesday at 2 PM"
    
    - **user_id**: User identifier
    - **command_text**: Natural language command
    - **audio_path**: Optional audio recording path
    """
    try:
        logger.info(f"Parsing natural language command for user {command.user_id}")
        
        # Parse natural language command
        parsed_reminder = _parse_reminder_from_text(
            text=command.command_text,
            user_id=command.user_id
        )
        
        logger.info(f"Parsed NL command for user {command.user_id}")
        
        return {
            "status": "success",
            "message": "Reminder parsed from natural language",
            "parsed_reminder": parsed_reminder,
            "original_command": command.command_text
        }
        
    except Exception as e:
        logger.error(f"Error parsing NL command: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/respond", response_model=dict)
async def process_reminder_response(response: ReminderResponse):
    """
    Process user's response to a reminder notification.
    
    Analyzes response using Pitt Corpus trained models to detect:
    - Confusion or memory issues
    - Confirmation or completion
    - Need for caregiver intervention
    
    - **reminder_id**: Reminder identifier
    - **user_id**: User identifier
    - **response_text**: User's text response
    - **audio_path**: Optional audio recording
    """
    try:
        # Get reminder details (TODO: from database)
        # For now, create mock reminder
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        reminder = Reminder(
            id=response.reminder_id,
            user_id=response.user_id,
            title="Sample Reminder",
            scheduled_time=datetime.now(),
            priority=ReminderPriority.MEDIUM,
            category="medication"
        )
        
        # Process response through adaptive scheduler
        result = scheduler.process_reminder_response(
            reminder=reminder,
            user_response=response.response_text,
            audio_path=response.audio_path
        )
        
        return {
            "status": "success",
            "message": "Response processed successfully",
            "analysis": result['analysis'],
            "interaction_type": result['analysis']['interaction_type'],
            "recommended_action": result['analysis']['recommended_action'],
            "caregiver_notified": result.get('action_result', {}).get('caregiver_notified', False),
            "cognitive_risk_score": result['analysis']['cognitive_risk_score']
        }
        
    except Exception as e:
        logger.error(f"Error processing response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/user/{user_id}", response_model=dict)
async def get_user_reminders(
    user_id: str,
    status_filter: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Get all reminders for a user.
    
    - **user_id**: User identifier
    - **status_filter**: Optional filter by status (active, completed, missed, etc.)
    - **category**: Optional filter by category
    """
    try:
        # Convert status_filter string to enum if provided
        status_enum = None
        if status_filter:
            try:
                status_enum = ReminderStatus(status_filter.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}. Valid values: active, completed, missed, snoozed, cancelled"
                )
        
        # Query database for user reminders
        try:
            db_service = get_db_service()
            reminders = await db_service.get_user_reminders(user_id, status_enum)
        except RuntimeError as db_error:
            logger.error(f"Database not connected: {db_error}")
            # Return empty result if DB not connected
            reminders = []
        
        # Apply category filter if specified
        if category:
            reminders = [r for r in reminders if r.get("category") == category]
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_reminders": len(reminders),
            "reminders": reminders
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reminders: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reminders: {str(e)}"
        )


@router.get("/behavior/{user_id}", response_model=dict)
async def get_user_behavior_pattern(
    user_id: str,
    reminder_id: Optional[str] = None,
    category: Optional[str] = None,
    days: int = 30
):
    """
    Get user's behavior patterns and analytics.
    
    Returns statistics on:
    - Confirmation rates
    - Confusion frequency
    - Optimal reminder times
    - Cognitive risk trends
    
    - **user_id**: User identifier
    - **reminder_id**: Optional specific reminder
    - **category**: Optional reminder category
    - **days**: Analysis period (default 30 days)
    """
    try:
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            reminder_id=reminder_id,
            category=category,
            days=days
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "pattern": pattern.dict()
        }
        
    except Exception as e:
        logger.error(f"Error getting behavior pattern: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/schedule/{reminder_id}", response_model=dict)
async def get_optimal_schedule(reminder_id: str):
    """
    Get optimal schedule recommendation for a reminder.
    
    Returns adaptive scheduling recommendations based on:
    - Historical user behavior
    - Optimal response times
    - Cognitive risk patterns
    
    - **reminder_id**: Reminder identifier
    """
    try:
        # TODO: Get reminder from database
        # reminder = db_service.get_reminder(reminder_id)
        
        # Mock reminder
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        reminder = Reminder(
            id=reminder_id,
            user_id="user123",
            title="Sample Reminder",
            scheduled_time=datetime.now(),
            priority=ReminderPriority.MEDIUM,
            category="medication"
        )
        
        schedule_info = scheduler.get_optimal_reminder_schedule(reminder)
        
        return {
            "status": "success",
            "reminder_id": reminder_id,
            "schedule": schedule_info
        }
        
    except Exception as e:
        logger.error(f"Error getting schedule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/update/{reminder_id}", response_model=dict)
async def update_reminder(reminder_id: str, updated_reminder: Reminder):
    """
    Update an existing reminder.
    
    - **reminder_id**: Reminder identifier
    - **updated_reminder**: Updated reminder data
    """
    try:
        updated_reminder.id = reminder_id
        updated_reminder.updated_at = datetime.now()
        
        # TODO: Update in database
        # db_service.update_reminder(updated_reminder.dict())
        
        logger.info(f"Updated reminder {reminder_id}")
        
        return {
            "status": "success",
            "message": "Reminder updated successfully",
            "reminder": updated_reminder.dict()
        }
        
    except Exception as e:
        logger.error(f"Error updating reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/delete/{reminder_id}", response_model=dict)
async def delete_reminder(reminder_id: str):
    """
    Delete a reminder.
    
    - **reminder_id**: Reminder identifier
    """
    try:
        # TODO: Delete from database
        # db_service.delete_reminder(reminder_id)
        
        logger.info(f"Deleted reminder {reminder_id}")
        
        return {
            "status": "success",
            "message": "Reminder deleted successfully",
            "reminder_id": reminder_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/snooze/{reminder_id}", response_model=dict)
async def snooze_reminder(reminder_id: str, delay_minutes: int = 15):
    """
    Snooze a reminder for specified duration.
    
    - **reminder_id**: Reminder identifier
    - **delay_minutes**: Minutes to delay (default 15)
    """
    try:
        # TODO: Get reminder from database
        from src.features.reminder_system.reminder_models import Reminder, ReminderPriority
        
        reminder = Reminder(
            id=reminder_id,
            user_id="user123",
            title="Sample Reminder",
            scheduled_time=datetime.now(),
            priority=ReminderPriority.MEDIUM,
            category="medication"
        )
        
        updated_reminder = scheduler.reschedule_reminder(
            reminder=reminder,
            delay_minutes=delay_minutes,
            reason="user_requested_snooze"
        )
        
        return {
            "status": "success",
            "message": f"Reminder snoozed for {delay_minutes} minutes",
            "new_scheduled_time": updated_reminder.scheduled_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error snoozing reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Caregiver endpoints

@router.get("/caregiver/alerts/{caregiver_id}", response_model=dict)
async def get_caregiver_alerts(caregiver_id: str, active_only: bool = True):
    """
    Get alerts for a caregiver.
    
    - **caregiver_id**: Caregiver identifier
    - **active_only**: Only return unresolved alerts (default True)
    """
    try:
        if active_only:
            alerts = caregiver_notifier.get_active_alerts(caregiver_id)
        else:
            # TODO: Get all alerts from database
            alerts = []
        
        return {
            "status": "success",
            "caregiver_id": caregiver_id,
            "total_alerts": len(alerts),
            "alerts": [alert.dict() for alert in alerts]
        }
        
    except Exception as e:
        logger.error(f"Error getting caregiver alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/caregiver/alerts/{alert_id}/acknowledge", response_model=dict)
async def acknowledge_alert(alert_id: str, caregiver_id: str):
    """
    Acknowledge a caregiver alert.
    
    - **alert_id**: Alert identifier
    - **caregiver_id**: Caregiver acknowledging the alert
    """
    try:
        success = caregiver_notifier.acknowledge_alert(alert_id, caregiver_id)
        
        if success:
            return {
                "status": "success",
                "message": "Alert acknowledged",
                "alert_id": alert_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to acknowledge alert"
            )
        
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/caregiver/alerts/{alert_id}/resolve", response_model=dict)
async def resolve_alert(alert_id: str, caregiver_id: str):
    """
    Mark an alert as resolved.
    
    - **alert_id**: Alert identifier
    - **caregiver_id**: Caregiver resolving the alert
    """
    try:
        success = caregiver_notifier.resolve_alert(alert_id, caregiver_id)
        
        if success:
            return {
                "status": "success",
                "message": "Alert resolved",
                "alert_id": alert_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to resolve alert"
            )
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/analytics/dashboard/{user_id}", response_model=dict)
async def get_user_dashboard(user_id: str, days: int = 7):
    """
    Get comprehensive dashboard analytics for a user.
    
    Returns:
    - Reminder statistics
    - Cognitive risk trends
    - Behavior patterns
    - Recent interactions
    
    - **user_id**: User identifier
    - **days**: Analysis period (default 7 days)
    """
    try:
        # Get behavior pattern
        pattern = behavior_tracker.get_user_behavior_pattern(
            user_id=user_id,
            days=days
        )
        
        # Calculate statistics
        total = pattern.total_reminders
        confirmation_rate = (pattern.confirmed_count / total) if total > 0 else 0
        confusion_rate = (pattern.confused_count / total) if total > 0 else 0
        
        dashboard = {
            "user_id": user_id,
            "period_days": days,
            "statistics": {
                "total_reminders": total,
                "confirmed": pattern.confirmed_count,
                "ignored": pattern.ignored_count,
                "delayed": pattern.delayed_count,
                "confused": pattern.confused_count,
                "confirmation_rate": confirmation_rate,
                "confusion_rate": confusion_rate
            },
            "cognitive_health": {
                "avg_risk_score": pattern.avg_cognitive_risk_score,
                "trend": pattern.confusion_trend,
                "escalation_recommended": pattern.escalation_recommended
            },
            "timing": {
                "optimal_hour": pattern.optimal_reminder_hour,
                "worst_hours": pattern.worst_response_hours,
                "avg_response_time_seconds": pattern.avg_response_time_seconds
            },
            "recommendations": {
                "frequency_multiplier": pattern.recommended_frequency_multiplier,
                "time_adjustment_minutes": pattern.recommended_time_adjustment_minutes
            }
        }
        
        return {
            "status": "success",
            "dashboard": dashboard
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/reports/weekly/{user_id}", response_model=dict)
async def get_weekly_report(
    user_id: str,
    end_date: Optional[str] = None,
    format: str = "json"
):
    """
    Generate comprehensive weekly risk monitoring report.
    
    Returns detailed analysis including:
    - Cognitive risk trends (daily and weekly)
    - Reminder completion statistics
    - Confusion patterns and memory issues
    - Caregiver alert frequency
    - Time-of-day performance analysis
    - Category-specific breakdowns
    - Actionable recommendations
    - Week-over-week comparison
    
    - **user_id**: Patient identifier
    - **end_date**: End date for report (ISO 8601 format, default: today)
    - **format**: Output format ('json' or 'pdf', default: 'json')
    """
    try:
        # Parse end_date if provided
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid end_date format. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
                )
        else:
            end_dt = None
        
        # Generate report
        report = report_generator.generate_weekly_report(
            user_id=user_id,
            end_date=end_dt
        )
        
        # Export to file if requested
        if format == "pdf":
            output_path = f"reports/weekly_{user_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
            report_generator.export_report_to_pdf(report, output_path)
            return {
                "status": "success",
                "message": "PDF report generated",
                "file_path": output_path,
                "report_summary": {
                    "user_id": report.user_id,
                    "period": f"{report.report_period_start} to {report.report_period_end}",
                    "risk_level": report.risk_level,
                    "intervention_needed": report.intervention_needed
                }
            }
        
        # Return JSON report
        return {
            "status": "success",
            "report": report.dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/reports/weekly/{user_id}/summary", response_model=dict)
async def get_weekly_report_summary(user_id: str):
    """
    Get brief summary of weekly report (for dashboard display).
    
    Returns key metrics only:
    - Risk level and score
    - Completion rate
    - Alert count
    - Intervention status
    
    - **user_id**: Patient identifier
    """
    try:
        report = report_generator.generate_weekly_report(user_id=user_id)
        
        return {
            "status": "success",
            "summary": {
                "user_id": report.user_id,
                "period": f"{report.report_period_start} to {report.report_period_end}",
                "risk_level": report.risk_level,
                "avg_cognitive_risk": report.avg_cognitive_risk_score,
                "risk_trend": report.risk_trend,
                "completion_rate": report.completion_rate,
                "total_alerts": report.total_alerts,
                "critical_alerts": report.critical_alerts,
                "intervention_needed": report.intervention_needed,
                "escalation_required": report.escalation_required,
                "top_recommendations": report.recommendations[:3]
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating report summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/complete/{reminder_id}", response_model=dict, status_code=status.HTTP_200_OK)
async def complete_reminder(reminder_id: str):
    """
    Mark a reminder as completed and handle repeat patterns.
    
    For daily/weekly reminders, this will:
    - Mark current instance as completed
    - Create next occurrence automatically
    
    - **reminder_id**: Reminder identifier
    """
    try:
        db_service = get_db_service()
        
        # Get the reminder
        reminder_data = await db_service.get_reminder(reminder_id)
        if not reminder_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reminder {reminder_id} not found"
            )
        
        # Mark as completed
        completion_time = datetime.now()
        await db_service.update_reminder(
            reminder_id,
            {
                "status": "completed",
                "completed_at": completion_time
            }
        )
        
        logger.info(f"Marked reminder {reminder_id} as completed")
        
        # Handle repeat patterns
        repeat_pattern = reminder_data.get("repeat_pattern")
        next_reminder_id = None
        next_scheduled_time = None
        
        if repeat_pattern in ["daily", "weekly"]:
            # Calculate next occurrence
            current_scheduled = reminder_data["scheduled_time"]
            if isinstance(current_scheduled, str):
                current_scheduled = datetime.fromisoformat(current_scheduled.replace('+00:00', ''))
            
            if repeat_pattern == "daily":
                next_scheduled_time = current_scheduled + timedelta(days=1)
            elif repeat_pattern == "weekly":
                next_scheduled_time = current_scheduled + timedelta(weeks=1)
            
            # Create next reminder instance
            from src.features.reminder_system.reminder_models import Reminder, ReminderPriority, ReminderStatus
            
            next_reminder = Reminder(
                id=f"reminder_{uuid.uuid4().hex[:12]}",
                user_id=reminder_data["user_id"],
                title=reminder_data["title"],
                description=reminder_data.get("description"),
                scheduled_time=next_scheduled_time,
                priority=ReminderPriority(reminder_data.get("priority", "medium")),
                category=reminder_data.get("category", "general"),
                repeat_pattern=repeat_pattern,
                repeat_interval_minutes=reminder_data.get("repeat_interval_minutes"),
                caregiver_ids=reminder_data.get("caregiver_ids", []),
                adaptive_scheduling_enabled=reminder_data.get("adaptive_scheduling_enabled", True),
                escalation_enabled=reminder_data.get("escalation_enabled", True),
                escalation_threshold_minutes=reminder_data.get("escalation_threshold_minutes", 30),
                notify_caregiver_on_miss=reminder_data.get("notify_caregiver_on_miss", True),
                status=ReminderStatus.ACTIVE
            )
            
            result = await db_service.create_reminder(next_reminder)
            next_reminder_id = result["id"]
            
            logger.info(f"Created next {repeat_pattern} reminder {next_reminder_id} scheduled for {next_scheduled_time}")
        
        return {
            "status": "success",
            "message": "Reminder completed successfully",
            "reminder_id": reminder_id,
            "completed_at": completion_time.isoformat(),
            "has_next_occurrence": next_reminder_id is not None,
            "next_reminder_id": next_reminder_id,
            "next_scheduled_time": next_scheduled_time.isoformat() if next_scheduled_time else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint for reminder system."""
    return {
        "status": "healthy",
        "service": "reminder_system",
        "timestamp": datetime.now().isoformat()
    }
