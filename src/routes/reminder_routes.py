"""
Reminder System API Routes

FastAPI endpoints for the Context-Aware Smart Reminder System.

Provides endpoints for:
- Creating, updating, and managing reminders
- Processing user responses
- Getting behavior analytics
- Caregiver notifications
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
from datetime import datetime
import logging

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

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/reminders", tags=["reminders"])

# Initialize components (these would typically come from dependency injection)
reminder_analyzer = PittBasedReminderAnalyzer()
behavior_tracker = BehaviorTracker()
scheduler = AdaptiveReminderScheduler(behavior_tracker, reminder_analyzer)
caregiver_notifier = CaregiverNotifier()
report_generator = WeeklyReportGenerator(behavior_tracker)


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
        # Set creation timestamps
        reminder.created_at = datetime.now()
        reminder.updated_at = datetime.now()
        
        # TODO: Save to database
        # db_service.save_reminder(reminder.dict())
        
        logger.info(f"Created reminder {reminder.id} for user {reminder.user_id}")
        
        return {
            "status": "success",
            "message": "Reminder created successfully",
            "reminder": reminder.dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/natural-language", response_model=dict)
async def create_reminder_from_natural_language(command: ReminderCommand):
    """
    Create reminder from natural language command using NLP.
    
    Examples:
    - "Remind me to take my tablets after lunch"
    - "Set medication reminder for 8 AM daily"
    - "Doctor appointment next Tuesday at 2 PM"
    
    - **user_id**: User identifier
    - **command_text**: Natural language command
    - **audio_path**: Optional audio recording path
    """
    try:
        # Parse natural language command
        # TODO: Implement NLP-based command parsing
        # This would use BERT to extract:
        # - Action (medication, appointment, meal, etc.)
        # - Time/date information
        # - Repetition pattern
        # - Priority indicators
        
        # For now, return a placeholder
        parsed_reminder = {
            "user_id": command.user_id,
            "title": "Extracted from: " + command.command_text,
            "description": command.command_text,
            "scheduled_time": datetime.now(),
            "category": "general",
            "priority": "medium"
        }
        
        logger.info(f"Parsed NL command for user {command.user_id}")
        
        return {
            "status": "success",
            "message": "Reminder created from natural language",
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
    status_filter: Optional[ReminderStatus] = None,
    category: Optional[str] = None
):
    """
    Get all reminders for a user.
    
    - **user_id**: User identifier
    - **status_filter**: Optional filter by status (active, completed, missed, etc.)
    - **category**: Optional filter by category
    """
    try:
        # TODO: Query database
        # reminders = db_service.get_user_reminders(user_id, status_filter, category)
        
        # Mock response
        reminders = []
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_reminders": len(reminders),
            "reminders": reminders
        }
        
    except Exception as e:
        logger.error(f"Error getting reminders: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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


@router.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint for reminder system."""
    return {
        "status": "healthy",
        "service": "reminder_system",
        "timestamp": datetime.now().isoformat()
    }
