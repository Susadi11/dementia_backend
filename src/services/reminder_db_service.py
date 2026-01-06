"""
Reminder Database Service

Enhanced database service specifically for reminder system persistence.
Handles reminder schedules, interactions, behavior analytics, and caregiver alerts.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json
from bson import ObjectId

from src.database import Database
from src.features.reminder_system.reminder_models import (
    Reminder, ReminderInteraction, ReminderStatus, CaregiverAlert
)

logger = logging.getLogger(__name__)


class ReminderDatabaseService:
    """Enhanced database service for reminder system data persistence."""

    def __init__(self):
        # Collections will be initialized lazily when accessed
        self._reminders_collection = None
        self._interactions_collection = None
        self._behavior_patterns_collection = None
        self._caregiver_alerts_collection = None

    @property
    def reminders_collection(self):
        """Lazy-load reminders collection."""
        if self._reminders_collection is None:
            self._reminders_collection = Database.get_collection("reminders")
        return self._reminders_collection

    @property
    def interactions_collection(self):
        """Lazy-load interactions collection."""
        if self._interactions_collection is None:
            self._interactions_collection = Database.get_collection("reminder_interactions")
        return self._interactions_collection

    @property
    def behavior_patterns_collection(self):
        """Lazy-load behavior patterns collection."""
        if self._behavior_patterns_collection is None:
            self._behavior_patterns_collection = Database.get_collection("user_behavior_patterns")
        return self._behavior_patterns_collection

    @property
    def caregiver_alerts_collection(self):
        """Lazy-load caregiver alerts collection."""
        if self._caregiver_alerts_collection is None:
            self._caregiver_alerts_collection = Database.get_collection("caregiver_alerts")
        return self._caregiver_alerts_collection

    # ===== REMINDER MANAGEMENT =====
    
    async def create_reminder(self, reminder: Reminder) -> Dict[str, Any]:
        """Create a new reminder in database."""
        try:
            # Generate ObjectId if not provided
            if not reminder.id:
                reminder.id = str(ObjectId())
                
            reminder_data = {
                "_id": reminder.id,
                "user_id": reminder.user_id,
                "title": reminder.title,
                "description": reminder.description,
                "scheduled_time": reminder.scheduled_time,
                "priority": reminder.priority.value,
                "category": reminder.category,
                "repeat_pattern": reminder.repeat_pattern,
                "repeat_interval_minutes": reminder.repeat_interval_minutes,
                "caregiver_ids": reminder.caregiver_ids,
                "adaptive_scheduling_enabled": reminder.adaptive_scheduling_enabled,
                "escalation_enabled": reminder.escalation_enabled,
                "escalation_threshold_minutes": reminder.escalation_threshold_minutes,
                "status": reminder.status.value,
                "notify_caregiver_on_miss": reminder.notify_caregiver_on_miss,
                "created_at": reminder.created_at,
                "updated_at": reminder.updated_at,
                "completed_at": reminder.completed_at
            }
            
            result = await self.reminders_collection.insert_one(reminder_data)
            
            logger.info(f"Created reminder: {reminder.id} for user {reminder.user_id}")
            
            return {
                "id": reminder.id,
                "inserted_id": str(result.inserted_id),
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Error creating reminder: {e}", exc_info=True)
            raise

    async def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """Get reminder by ID."""
        try:
            reminder = await self.reminders_collection.find_one({"_id": reminder_id})
            
            if reminder:
                # Convert ObjectId to string for JSON serialization
                reminder["id"] = str(reminder.pop("_id"))
                logger.info(f"Retrieved reminder: {reminder_id}")
                return reminder
            else:
                logger.warning(f"Reminder not found: {reminder_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving reminder {reminder_id}: {e}")
            return None

    async def update_reminder(self, reminder_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update reminder data."""
        try:
            update_data["updated_at"] = datetime.now()

            result = await self.reminders_collection.update_one(
                {"_id": reminder_id},
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                logger.info(f"Updated reminder: {reminder_id}")
                return {"id": reminder_id, "modified": result.modified_count > 0}
            else:
                logger.warning(f"Reminder not found for update: {reminder_id}")
                return {"id": reminder_id, "modified": False, "error": "not_found"}
                
        except Exception as e:
            logger.error(f"Error updating reminder {reminder_id}: {e}")
            raise

    async def get_user_reminders(
        self, 
        user_id: str, 
        status: Optional[ReminderStatus] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all reminders for a user with optional status filter."""
        try:
            query = {"user_id": user_id}
            if status:
                query["status"] = status.value

            cursor = self.reminders_collection.find(query).limit(limit).sort("scheduled_time", 1)
            
            reminders = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                reminders.append(doc)
                
            logger.info(f"Retrieved {len(reminders)} reminders for user {user_id}")
            return reminders
            
        except Exception as e:
            logger.error(f"Error getting reminders for user {user_id}: {e}")
            return []

    async def get_due_reminders(self, time_window_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get reminders due within the specified time window."""
        try:
            current_time = datetime.now()
            window_end = current_time + timedelta(minutes=time_window_minutes)
            
            query = {
                "scheduled_time": {
                    "$gte": current_time,
                    "$lte": window_end
                },
                "status": "active"
            }
            
            cursor = self.reminders_collection.find(query).sort("scheduled_time", 1)
            
            reminders = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                reminders.append(doc)
                
            logger.info(f"Found {len(reminders)} due reminders between {current_time} and {window_end}")
            return reminders
            
        except Exception as e:
            logger.error(f"Error getting due reminders: {e}")
            return []

    async def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder."""
        try:
            result = await self.reminders_collection.delete_one({"_id": reminder_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted reminder: {reminder_id}")
                return True
            else:
                logger.warning(f"Reminder not found for deletion: {reminder_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting reminder {reminder_id}: {e}")
            return False

    # ===== INTERACTION TRACKING =====
    
    async def log_reminder_interaction(self, interaction: ReminderInteraction) -> Dict[str, Any]:
        """Log a reminder interaction."""
        interaction_data = {
            "id": f"interaction_{datetime.now().timestamp()}",
            "reminder_id": interaction.reminder_id,
            "user_id": interaction.user_id,
            "response_text": interaction.response_text,
            "response_time_seconds": interaction.response_time_seconds,
            "interaction_type": interaction.interaction_type.value,
            "cognitive_risk_score": interaction.cognitive_risk_score,
            "confusion_detected": interaction.confusion_detected,
            "memory_issue_detected": interaction.memory_issue_detected,
            "recommended_action": interaction.recommended_action,
            "caregiver_alert_needed": interaction.caregiver_alert_needed,
            "timestamp": interaction.timestamp.isoformat()
        }
        
        logger.info(f"Logging interaction for reminder {interaction.reminder_id}")
        return interaction_data

    async def get_user_interactions(
        self, 
        user_id: str, 
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get user interactions within specified time period."""
        since_date = datetime.now() - timedelta(days=days_back)
        logger.info(f"Getting interactions for user {user_id} since {since_date}")
        # Mock implementation - replace with actual database query
        return []

    # ===== BEHAVIOR ANALYTICS =====
    
    async def update_behavior_pattern(
        self, 
        user_id: str, 
        pattern_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user behavior pattern analysis."""
        pattern_data.update({
            "user_id": user_id,
            "last_updated": datetime.now().isoformat(),
            "analysis_period_days": pattern_data.get("analysis_period_days", 30)
        })
        
        logger.info(f"Updating behavior pattern for user {user_id}")
        return pattern_data

    async def get_behavior_pattern(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get latest behavior pattern analysis for user."""
        logger.info(f"Getting behavior pattern for user {user_id}")
        # Mock implementation - replace with actual database query
        return {
            "user_id": user_id,
            "confirmation_rate": 0.85,
            "average_response_time": 23.5,
            "cognitive_risk_trend": "stable",
            "optimal_reminder_hours": [8, 12, 18]
        }

    # ===== CAREGIVER ALERTS =====
    
    async def create_caregiver_alert(self, alert: CaregiverAlert) -> Dict[str, Any]:
        """Create a caregiver alert."""
        alert_data = {
            "id": alert.id,
            "caregiver_id": alert.caregiver_id,
            "user_id": alert.user_id,
            "reminder_id": alert.reminder_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "message": alert.message,
            "context": json.dumps(alert.context),
            "acknowledged": alert.acknowledged,
            "resolved": alert.resolved,
            "created_at": alert.created_at.isoformat()
        }
        
        logger.info(f"Creating caregiver alert: {alert.id}")
        return alert_data

    async def get_caregiver_alerts(
        self, 
        caregiver_id: str, 
        unresolved_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get alerts for a caregiver."""
        logger.info(f"Getting alerts for caregiver {caregiver_id}, unresolved only: {unresolved_only}")
        # Mock implementation - replace with actual database query
        return []

    async def acknowledge_alert(self, alert_id: str, caregiver_id: str) -> bool:
        """Mark alert as acknowledged."""
        logger.info(f"Acknowledging alert {alert_id} by caregiver {caregiver_id}")
        return True

    async def resolve_alert(self, alert_id: str, caregiver_id: str, resolution_notes: str) -> bool:
        """Mark alert as resolved with notes."""
        logger.info(f"Resolving alert {alert_id} with notes: {resolution_notes}")
        return True

    # ===== ANALYTICS QUERIES =====
    
    async def get_reminder_analytics(
        self, 
        user_id: str, 
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive reminder analytics for user."""
        logger.info(f"Getting analytics for user {user_id}, {days_back} days back")
        
        # Mock analytics data - replace with actual calculations
        return {
            "total_reminders": 45,
            "completion_rate": 0.78,
            "average_response_time": 28.5,
            "confusion_incidents": 3,
            "cognitive_decline_indicators": 2,
            "caregiver_alerts_count": 1,
            "optimal_times": [8, 13, 19],
            "category_performance": {
                "medication": 0.92,
                "meal": 0.71,
                "appointment": 0.85
            }
        }

    async def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics."""
        logger.info("Getting system-wide analytics")
        
        return {
            "total_active_users": 150,
            "total_reminders_today": 567,
            "completion_rate_today": 0.82,
            "active_alerts": 12,
            "high_risk_users": 8
        }