"""
Real-Time Reminder Processing Engine

Handles live reminder delivery, response processing, and real-time analytics.
Integrates with WebSocket connections for instant notifications.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
from dataclasses import asdict

from fastapi import WebSocket
from src.features.reminder_system import (
    PittBasedReminderAnalyzer, AdaptiveReminderScheduler, 
    BehaviorTracker, Reminder, ReminderInteraction
)
from src.features.reminder_system.caregiver_notifier import CaregiverNotifier
from src.services.reminder_db_service import ReminderDatabaseService

logger = logging.getLogger(__name__)


class RealTimeReminderEngine:
    """
    Real-time processing engine for the reminder system.
    
    Handles:
    - Live reminder delivery via WebSocket
    - Real-time response processing
    - Instant caregiver notifications
    - Background scheduling tasks
    """
    
    def __init__(self):
        self.db_service = ReminderDatabaseService()
        self.reminder_analyzer = PittBasedReminderAnalyzer()
        self.behavior_tracker = BehaviorTracker()
        self.scheduler = AdaptiveReminderScheduler(
            self.behavior_tracker, 
            self.reminder_analyzer
        )
        self.caregiver_notifier = CaregiverNotifier()
        
        # Active WebSocket connections
        self.user_connections: Dict[str, WebSocket] = {}
        self.caregiver_connections: Dict[str, WebSocket] = {}
        
        # Background task control
        self.is_running = False
        self.reminder_check_interval = 30  # seconds
        
    async def start_engine(self):
        """Start the real-time processing engine."""
        if self.is_running:
            logger.warning("Engine is already running")
            return
            
        self.is_running = True
        logger.info("Starting real-time reminder engine")
        
        # Start background tasks
        asyncio.create_task(self._reminder_scheduler_task())
        asyncio.create_task(self._behavior_analysis_task())
        asyncio.create_task(self._cleanup_task())
    
    async def stop_engine(self):
        """Stop the real-time processing engine."""
        self.is_running = False
        logger.info("Stopping real-time reminder engine")
    
    # ===== WEBSOCKET CONNECTION MANAGEMENT =====
    
    async def connect_user(self, user_id: str, websocket: WebSocket):
        """Connect a user's WebSocket for real-time notifications."""
        await websocket.accept()
        self.user_connections[user_id] = websocket
        logger.info(f"User {user_id} connected to real-time engine")
        
        # Send initial status
        await self._send_user_message(user_id, {
            "type": "connection_established",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to smart reminder system"
        })
    
    async def disconnect_user(self, user_id: str):
        """Disconnect a user's WebSocket."""
        if user_id in self.user_connections:
            del self.user_connections[user_id]
            logger.info(f"User {user_id} disconnected from real-time engine")
    
    async def connect_caregiver(self, caregiver_id: str, websocket: WebSocket):
        """Connect a caregiver's WebSocket for real-time alerts."""
        await websocket.accept()
        self.caregiver_connections[caregiver_id] = websocket
        logger.info(f"Caregiver {caregiver_id} connected to real-time engine")
        
        # Send pending alerts
        await self._send_pending_alerts(caregiver_id)
    
    async def disconnect_caregiver(self, caregiver_id: str):
        """Disconnect a caregiver's WebSocket."""
        if caregiver_id in self.caregiver_connections:
            del self.caregiver_connections[caregiver_id]
            logger.info(f"Caregiver {caregiver_id} disconnected from real-time engine")
    
    # ===== REAL-TIME REMINDER DELIVERY =====
    
    async def deliver_reminder(self, reminder: Reminder):
        """Deliver a reminder in real-time via WebSocket."""
        user_id = reminder.user_id
        
        if user_id not in self.user_connections:
            logger.warning(f"No active connection for user {user_id}, cannot deliver reminder")
            return False
        
        reminder_message = {
            "type": "reminder",
            "reminder_id": reminder.id,
            "title": reminder.title,
            "description": reminder.description,
            "priority": reminder.priority.value,
            "category": reminder.category,
            "scheduled_time": reminder.scheduled_time.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            success = await self._send_user_message(user_id, reminder_message)
            if success:
                logger.info(f"Reminder {reminder.id} delivered to user {user_id}")
                
                # Update reminder status
                await self.db_service.update_reminder(
                    reminder.id, 
                    {"status": "delivered", "delivered_at": datetime.now().isoformat()}
                )
            return success
        except Exception as e:
            logger.error(f"Failed to deliver reminder {reminder.id}: {e}")
            return False
    
    async def process_user_response(
        self, 
        user_id: str, 
        reminder_id: str, 
        response_text: str,
        response_time_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process a user's response to a reminder in real-time."""
        logger.info(f"Processing response from user {user_id} for reminder {reminder_id}")
        
        # Get reminder context
        reminder_data = await self.db_service.get_reminder(reminder_id)
        if not reminder_data:
            logger.error(f"Reminder {reminder_id} not found")
            return {"error": "Reminder not found"}
        
        # Analyze response using Pitt-based analyzer
        try:
            analysis_result = await self._analyze_response(
                response_text, reminder_data, response_time_seconds
            )
            
            # Create interaction record
            interaction = ReminderInteraction(
                reminder_id=reminder_id,
                user_id=user_id,
                response_text=response_text,
                response_time_seconds=response_time_seconds or 0.0,
                interaction_type=analysis_result["interaction_type"],
                cognitive_risk_score=analysis_result["cognitive_risk_score"],
                confusion_detected=analysis_result["confusion_detected"],
                memory_issue_detected=analysis_result["memory_issue_detected"],
                recommended_action=analysis_result["recommended_action"],
                caregiver_alert_needed=analysis_result["caregiver_alert_needed"]
            )
            
            # Log interaction
            await self.db_service.log_reminder_interaction(interaction)
            
            # Process adaptive scheduling
            await self._process_adaptive_scheduling(reminder_data, interaction)
            
            # Handle caregiver alerts if needed
            if analysis_result["caregiver_alert_needed"]:
                await self._send_caregiver_alerts(reminder_data, interaction, analysis_result)
            
            # Send real-time feedback to user
            await self._send_response_feedback(user_id, analysis_result)
            
            return {
                "success": True,
                "analysis": analysis_result,
                "next_action": analysis_result["recommended_action"]
            }
            
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
            return {"error": f"Failed to process response: {str(e)}"}
    
    # ===== BACKGROUND TASKS =====
    
    async def _reminder_scheduler_task(self):
        """Background task that checks for due reminders."""
        while self.is_running:
            try:
                # Get reminders due in the next 5 minutes
                due_reminders = await self.db_service.get_due_reminders(5)
                
                for reminder_data in due_reminders:
                    # Convert to Reminder object and deliver
                    reminder = self._dict_to_reminder(reminder_data)
                    await self.deliver_reminder(reminder)
                
                await asyncio.sleep(self.reminder_check_interval)
                
            except Exception as e:
                logger.error(f"Error in reminder scheduler task: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _behavior_analysis_task(self):
        """Background task for updating user behavior patterns."""
        while self.is_running:
            try:
                # Update behavior patterns every hour
                # This would get all active users and update their patterns
                logger.info("Running behavior pattern analysis...")
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in behavior analysis task: {e}")
                await asyncio.sleep(1800)  # 30 minutes on error
    
    async def _cleanup_task(self):
        """Background task for cleanup and maintenance."""
        while self.is_running:
            try:
                # Clean up old data, closed connections, etc.
                await self._cleanup_closed_connections()
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(600)  # 10 minutes on error
    
    # ===== HELPER METHODS =====
    
    async def _send_user_message(self, user_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a user via WebSocket."""
        if user_id not in self.user_connections:
            return False
        
        try:
            websocket = self.user_connections[user_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to user {user_id}: {e}")
            # Remove broken connection
            await self.disconnect_user(user_id)
            return False
    
    async def _send_caregiver_message(self, caregiver_id: str, message: Dict[str, Any]) -> bool:
        """Send a message to a caregiver via WebSocket."""
        if caregiver_id not in self.caregiver_connections:
            return False
        
        try:
            websocket = self.caregiver_connections[caregiver_id]
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to caregiver {caregiver_id}: {e}")
            # Remove broken connection
            await self.disconnect_caregiver(caregiver_id)
            return False
    
    async def _analyze_response(
        self, 
        response_text: str, 
        reminder_data: Dict[str, Any],
        response_time_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze user response using Pitt-based analyzer."""
        return self.reminder_analyzer.analyze_reminder_response(
            response_text, 
            reminder_data,
            response_time_seconds
        )
    
    async def _process_adaptive_scheduling(
        self, 
        reminder_data: Dict[str, Any], 
        interaction: ReminderInteraction
    ):
        """Process adaptive scheduling based on interaction."""
        # Update behavior tracker
        self.behavior_tracker.log_interaction(interaction)
        
        # Get updated scheduling recommendations
        reminder = self._dict_to_reminder(reminder_data)
        schedule_update = self.scheduler.process_reminder_response(
            reminder, interaction.response_text, interaction.response_time_seconds
        )
        
        # Update reminder in database if needed
        if schedule_update.get("reschedule_needed"):
            await self.db_service.update_reminder(
                reminder.id,
                {
                    "scheduled_time": schedule_update["next_scheduled_time"],
                    "frequency_multiplier": schedule_update.get("frequency_multiplier", 1.0)
                }
            )
    
    async def _send_caregiver_alerts(
        self, 
        reminder_data: Dict[str, Any], 
        interaction: ReminderInteraction,
        analysis_result: Dict[str, Any]
    ):
        """Send alerts to caregivers if needed."""
        caregiver_ids = json.loads(reminder_data.get("caregiver_ids", "[]"))
        
        for caregiver_id in caregiver_ids:
            alert_message = {
                "type": "user_alert",
                "user_id": interaction.user_id,
                "reminder_id": interaction.reminder_id,
                "alert_type": analysis_result.get("alert_type", "confusion"),
                "severity": analysis_result.get("severity", "medium"),
                "message": analysis_result.get("caregiver_message", "User needs attention"),
                "timestamp": datetime.now().isoformat(),
                "interaction_summary": {
                    "response": interaction.response_text,
                    "cognitive_risk": interaction.cognitive_risk_score,
                    "confusion_detected": interaction.confusion_detected
                }
            }
            
            await self._send_caregiver_message(caregiver_id, alert_message)
    
    async def _send_response_feedback(self, user_id: str, analysis_result: Dict[str, Any]):
        """Send real-time feedback to user based on response analysis."""
        feedback_message = {
            "type": "response_feedback",
            "message": analysis_result.get("user_feedback", "Response recorded"),
            "next_action": analysis_result.get("recommended_action"),
            "timestamp": datetime.now().isoformat()
        }
        
        await self._send_user_message(user_id, feedback_message)
    
    async def _send_pending_alerts(self, caregiver_id: str):
        """Send pending alerts to newly connected caregiver."""
        alerts = await self.db_service.get_caregiver_alerts(caregiver_id, unresolved_only=True)
        
        for alert in alerts:
            await self._send_caregiver_message(caregiver_id, {
                "type": "pending_alert",
                **alert
            })
    
    async def _cleanup_closed_connections(self):
        """Clean up closed WebSocket connections."""
        # Clean user connections
        closed_users = []
        for user_id, ws in self.user_connections.items():
            if ws.client_state.value == 3:  # CLOSED
                closed_users.append(user_id)
        
        for user_id in closed_users:
            del self.user_connections[user_id]
            logger.info(f"Cleaned up closed connection for user {user_id}")
        
        # Clean caregiver connections
        closed_caregivers = []
        for caregiver_id, ws in self.caregiver_connections.items():
            if ws.client_state.value == 3:  # CLOSED
                closed_caregivers.append(caregiver_id)
        
        for caregiver_id in closed_caregivers:
            del self.caregiver_connections[caregiver_id]
            logger.info(f"Cleaned up closed connection for caregiver {caregiver_id}")
    
    def _dict_to_reminder(self, reminder_data: Dict[str, Any]) -> Reminder:
        """Convert dictionary to Reminder object."""
        # This would be implemented based on your Reminder model structure
        # Mock implementation for now
        from src.features.reminder_system.reminder_models import ReminderStatus, ReminderPriority
        
        # Handle scheduled_time - could be string or datetime
        scheduled_time = reminder_data["scheduled_time"]
        if isinstance(scheduled_time, str):
            scheduled_time = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
        elif not isinstance(scheduled_time, datetime):
            scheduled_time = datetime.now()  # Fallback
        
        return Reminder(
            id=reminder_data["id"],
            user_id=reminder_data["user_id"],
            title=reminder_data["title"],
            description=reminder_data.get("description", ""),
            scheduled_time=scheduled_time,
            priority=ReminderPriority(reminder_data.get("priority", "medium")),
            category=reminder_data.get("category", "general"),
            repeat_pattern=reminder_data.get("repeat_pattern"),
            caregiver_ids=json.loads(reminder_data.get("caregiver_ids", "[]")),
            adaptive_scheduling_enabled=reminder_data.get("adaptive_scheduling_enabled", True),
            escalation_enabled=reminder_data.get("escalation_enabled", True),
            status=ReminderStatus(reminder_data.get("status", "active"))
        )