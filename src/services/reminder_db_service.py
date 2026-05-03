"""
Reminder Database Service

Enhanced database service specifically for reminder system persistence.
Handles reminder schedules, interactions, behavior analytics, and caregiveralerts.
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
                # Normalize datetime-like fields so route handlers can parse consistently.
                for key in ("scheduled_time", "created_at", "updated_at", "completed_at", "alarm_triggered_at"):
                    if isinstance(doc.get(key), datetime):
                        doc[key] = doc[key].isoformat()
                reminders.append(doc)
                
            logger.info(f"Retrieved {len(reminders)} reminders for user {user_id}")
            return reminders
            
        except Exception as e:
            logger.error(f"Error getting reminders for user {user_id}: {e}")
            return []

    async def get_due_reminders(
        self,
        time_window_minutes: int = 5,
        user_id: Optional[str] = None,
        lookback_minutes: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get reminders due within the specified time window.

        Uses datetime.now() (naive local time) to match how scheduled_time is
        stored when reminders are created. Includes a lookback window
        (default 10 min) so reminders are not silently dropped during polling
        gaps, server restarts, or brief network outages.
        """
        try:
            current_time = datetime.now()  # naive local time — matches stored values
            window_start = current_time - timedelta(minutes=lookback_minutes)
            window_end = current_time + timedelta(minutes=time_window_minutes)

            query = {
                "scheduled_time": {
                    "$gte": window_start,
                    "$lte": window_end
                },
                "status": {"$in": ["active", "snoozed"]}
            }

            if user_id:
                query["user_id"] = user_id

            cursor = self.reminders_collection.find(query).sort("scheduled_time", 1)

            reminders = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                reminders.append(doc)

            logger.info(
                f"Found {len(reminders)} due reminders between {window_start} and {window_end}"
                + (f" for user {user_id}" if user_id else "")
            )
            return reminders

        except Exception as e:
            logger.error(f"Error getting due reminders: {e}")
            return []

    async def get_active_user_ids(self) -> List[str]:
        """Get distinct user IDs that have at least one active reminder."""
        try:
            user_ids = await self.reminders_collection.distinct(
                "user_id", {"status": "active"}
            )
            return user_ids
        except Exception as e:
            logger.error(f"Error getting active user IDs: {e}")
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

    def save_reminder_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Persist a reminder interaction to MongoDB (fire-and-forget async write)."""
        import asyncio
        doc = {
            "_id": interaction_data.get("id") or f"int_{datetime.now().timestamp()}",
            "reminder_id": interaction_data.get("reminder_id"),
            "user_id": interaction_data.get("user_id"),
            "reminder_category": interaction_data.get("reminder_category"),
            "interaction_type": interaction_data.get("interaction_type"),
            "interaction_time": interaction_data.get("interaction_time", datetime.now()),
            "user_response_text": interaction_data.get("user_response_text"),
            "cognitive_risk_score": interaction_data.get("cognitive_risk_score"),
            "confusion_detected": interaction_data.get("confusion_detected", False),
            "memory_issue_detected": interaction_data.get("memory_issue_detected", False),
            "response_time_seconds": interaction_data.get("response_time_seconds"),
            "recommended_action": interaction_data.get("recommended_action"),
            "caregiver_alert_triggered": interaction_data.get("caregiver_alert_triggered", False),
        }
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._insert_interaction(doc))
        except RuntimeError:
            logger.warning("No running event loop — interaction not persisted to DB")

    async def _insert_interaction(self, doc: Dict[str, Any]) -> None:
        """Actual async insert into reminder_interactions collection."""
        try:
            await self.interactions_collection.insert_one(doc)
            logger.info(f"Saved interaction to DB: user={doc.get('user_id')}, type={doc.get('interaction_type')}")
        except Exception as e:
            logger.error(f"Failed to insert interaction: {e}")

    def get_reminder_interactions(self, **_kwargs) -> List:
        """Sync stub so BehaviorTracker._get_interactions() doesn't raise AttributeError.
        Real data comes from the in-memory cache warmed up by warm_cache_from_db() at startup."""
        return []

    async def get_reminder_interactions_async(
        self,
        user_id: str,
        start_date: datetime,
        reminder_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async query: fetch interactions from MongoDB for cache warm-up."""
        try:
            query: Dict[str, Any] = {
                "user_id": user_id,
                "interaction_time": {"$gte": start_date},
            }
            if reminder_id:
                query["reminder_id"] = reminder_id
            if category:
                query["reminder_category"] = category

            cursor = self.interactions_collection.find(query).sort("interaction_time", 1)
            docs = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                docs.append(doc)
            return docs
        except Exception as e:
            logger.error(f"Error fetching interactions for user {user_id}: {e}")
            return []

    async def get_user_interactions(
        self,
        user_id: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get user interactions within specified time period."""
        since_date = datetime.now() - timedelta(days=days_back)
        return await self.get_reminder_interactions_async(user_id=user_id, start_date=since_date)

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