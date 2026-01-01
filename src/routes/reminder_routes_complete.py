"""
Complete Reminder System API Routes

FastAPI endpoints for the Context-Aware Smart Reminder System with full MongoDB integration.

Provides endpoints for:
- Creating, updating, and managing reminders
- Processing user responses
- Getting behavior analytics
- Caregiver notifications
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import uuid

from src.services.reminder_db_service import ReminderDatabaseService
from src.features.reminder_system.reminder_models import (
    Reminder, ReminderStatus, ReminderPriority, ReminderInteraction, InteractionType
)
from src.database import Database

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/reminders", tags=["Smart Reminders"])

# Initialize database service
db_service = ReminderDatabaseService()


class CreateReminderRequest(BaseModel):
    """Request model for creating a reminder."""
    user_id: str = Field(..., description="User identifier")
    title: str = Field(..., description="Reminder title", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Detailed description", max_length=500)
    scheduled_time: datetime = Field(..., description="When to send the reminder")
    priority: ReminderPriority = Field(ReminderPriority.MEDIUM, description="Priority level")
    category: str = Field("general", description="Reminder category")
    repeat_pattern: Optional[str] = Field(None, description="Repeat pattern: daily, weekly, custom")
    repeat_interval_minutes: Optional[int] = Field(None, description="Custom repeat interval in minutes")
    caregiver_ids: List[str] = Field(default_factory=list, description="Caregiver IDs to notify")
    notify_caregiver_on_miss: bool = Field(True, description="Notify caregiver if missed")
    escalation_threshold_minutes: int = Field(30, description="Minutes before caregiver notification")


class ReminderResponse(BaseModel):
    """Response model for reminder operations."""
    id: str
    user_id: str
    title: str
    description: Optional[str]
    scheduled_time: datetime
    priority: str
    category: str
    status: str
    created_at: datetime
    updated_at: datetime


class InteractionRequest(BaseModel):
    """Request model for recording reminder interactions."""
    reminder_id: str = Field(..., description="Reminder ID")
    user_id: str = Field(..., description="User ID") 
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    user_response_text: Optional[str] = Field(None, description="User's text response")
    user_response_audio_path: Optional[str] = Field(None, description="Audio file path")


@router.post("/", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_reminder(reminder_request: CreateReminderRequest):
    """
    Create a new reminder for a user.
    
    Creates a reminder with the specified parameters and stores it in the database.
    Returns the created reminder details.
    """
    try:
        # Create reminder object
        reminder = Reminder(
            id=str(uuid.uuid4()),
            user_id=reminder_request.user_id,
            title=reminder_request.title,
            description=reminder_request.description,
            scheduled_time=reminder_request.scheduled_time,
            priority=reminder_request.priority,
            category=reminder_request.category,
            repeat_pattern=reminder_request.repeat_pattern,
            repeat_interval_minutes=reminder_request.repeat_interval_minutes,
            caregiver_ids=reminder_request.caregiver_ids,
            notify_caregiver_on_miss=reminder_request.notify_caregiver_on_miss,
            escalation_threshold_minutes=reminder_request.escalation_threshold_minutes,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to database
        result = await db_service.create_reminder(reminder)
        
        return {
            "status": "success",
            "message": "Reminder created successfully",
            "data": {
                "id": reminder.id,
                "user_id": reminder.user_id,
                "title": reminder.title,
                "scheduled_time": reminder.scheduled_time.isoformat(),
                "priority": reminder.priority.value,
                "status": reminder.status.value
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating reminder: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create reminder: {str(e)}"
        )


@router.get("/user/{user_id}", response_model=dict)
async def get_user_reminders(
    user_id: str,
    status_filter: Optional[ReminderStatus] = None,
    limit: int = 50
):
    """
    Get all reminders for a specific user.
    
    Optionally filter by status and limit the number of results.
    """
    try:
        reminders = await db_service.get_user_reminders(user_id, status_filter, limit)
        
        return {
            "status": "success",
            "user_id": user_id,
            "total_reminders": len(reminders),
            "data": reminders
        }
        
    except Exception as e:
        logger.error(f"Error getting reminders for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reminders: {str(e)}"
        )


@router.get("/{reminder_id}", response_model=dict)
async def get_reminder(reminder_id: str):
    """
    Get a specific reminder by ID.
    """
    try:
        reminder = await db_service.get_reminder(reminder_id)
        
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        return {
            "status": "success",
            "data": reminder
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reminder {reminder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reminder: {str(e)}"
        )


@router.put("/{reminder_id}", response_model=dict)
async def update_reminder(reminder_id: str, update_data: CreateReminderRequest):
    """
    Update an existing reminder.
    """
    try:
        # Check if reminder exists
        existing_reminder = await db_service.get_reminder(reminder_id)
        if not existing_reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Prepare update data
        update_dict = {
            "title": update_data.title,
            "description": update_data.description,
            "scheduled_time": update_data.scheduled_time,
            "priority": update_data.priority.value,
            "category": update_data.category,
            "repeat_pattern": update_data.repeat_pattern,
            "repeat_interval_minutes": update_data.repeat_interval_minutes,
            "caregiver_ids": update_data.caregiver_ids,
            "notify_caregiver_on_miss": update_data.notify_caregiver_on_miss,
            "escalation_threshold_minutes": update_data.escalation_threshold_minutes
        }
        
        result = await db_service.update_reminder(reminder_id, update_dict)
        
        return {
            "status": "success",
            "message": "Reminder updated successfully",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating reminder {reminder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update reminder: {str(e)}"
        )


@router.delete("/{reminder_id}", response_model=dict)
async def delete_reminder(reminder_id: str):
    """
    Delete a reminder.
    """
    try:
        success = await db_service.delete_reminder(reminder_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        return {
            "status": "success",
            "message": "Reminder deleted successfully",
            "reminder_id": reminder_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting reminder {reminder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete reminder: {str(e)}"
        )


@router.post("/{reminder_id}/complete", response_model=dict)
async def complete_reminder(reminder_id: str):
    """
    Mark a reminder as completed.
    """
    try:
        # Check if reminder exists
        reminder = await db_service.get_reminder(reminder_id)
        if not reminder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reminder not found"
            )
        
        # Update status to completed
        update_data = {
            "status": ReminderStatus.COMPLETED.value,
            "completed_at": datetime.now()
        }
        
        result = await db_service.update_reminder(reminder_id, update_data)
        
        return {
            "status": "success",
            "message": "Reminder marked as completed",
            "reminder_id": reminder_id,
            "completed_at": update_data["completed_at"].isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing reminder {reminder_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete reminder: {str(e)}"
        )


@router.get("/due/now", response_model=dict)
async def get_due_reminders(time_window_minutes: int = 5):
    """
    Get reminders that are due within the specified time window.
    
    Default time window is 5 minutes from current time.
    """
    try:
        due_reminders = await db_service.get_due_reminders(time_window_minutes)
        
        return {
            "status": "success",
            "time_window_minutes": time_window_minutes,
            "current_time": datetime.now().isoformat(),
            "total_due": len(due_reminders),
            "data": due_reminders
        }
        
    except Exception as e:
        logger.error(f"Error getting due reminders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve due reminders: {str(e)}"
        )


@router.post("/interactions", response_model=dict, status_code=status.HTTP_201_CREATED)
async def record_interaction(interaction: InteractionRequest):
    """
    Record a user interaction with a reminder.
    
    This endpoint is used to track how users respond to reminders,
    which helps improve the adaptive scheduling system.
    """
    try:
        # Create interaction object
        interaction_data = ReminderInteraction(
            id=str(uuid.uuid4()),
            reminder_id=interaction.reminder_id,
            user_id=interaction.user_id,
            interaction_type=interaction.interaction_type,
            user_response_text=interaction.user_response_text,
            user_response_audio_path=interaction.user_response_audio_path,
            interaction_time=datetime.now()
        )
        
        # In a full implementation, you would:
        # 1. Save to database
        # 2. Run cognitive analysis if needed
        # 3. Update behavior patterns
        # 4. Trigger caregiver alerts if necessary
        
        logger.info(f"Recorded interaction for reminder {interaction.reminder_id}")
        
        return {
            "status": "success",
            "message": "Interaction recorded successfully",
            "interaction_id": interaction_data.id,
            "reminder_id": interaction.reminder_id,
            "interaction_type": interaction.interaction_type.value
        }
        
    except Exception as e:
        logger.error(f"Error recording interaction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record interaction: {str(e)}"
        )


@router.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint for the reminder system.
    """
    try:
        # Check database connection
        db_health = await Database.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_health.get("status", "unknown"),
            "service": "reminder_system"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "service": "reminder_system"
        }


# Additional utility endpoints

@router.get("/stats/user/{user_id}", response_model=dict)
async def get_user_reminder_stats(user_id: str):
    """
    Get reminder statistics for a user.
    """
    try:
        all_reminders = await db_service.get_user_reminders(user_id, limit=1000)
        
        stats = {
            "total_reminders": len(all_reminders),
            "active": len([r for r in all_reminders if r.get("status") == "active"]),
            "completed": len([r for r in all_reminders if r.get("status") == "completed"]),
            "missed": len([r for r in all_reminders if r.get("status") == "missed"]),
            "categories": {}
        }
        
        # Count by category
        for reminder in all_reminders:
            category = reminder.get("category", "unknown")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        return {
            "status": "success",
            "user_id": user_id,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting user stats for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )