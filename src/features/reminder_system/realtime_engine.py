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
        
        # Alarm acknowledgment tracking
        # Format: {reminder_id: {"user_id": str, "triggered_at": datetime, "repeat_count": int, "last_repeat_at": datetime}}
        self.active_alarms: Dict[str, Dict[str, Any]] = {}
        
        # Alarm configuration
        self.alarm_timeout_seconds = 180  # Wait 3 minutes between each alarm attempt
        self.default_escalation_threshold_minutes = 10  # Default escalation threshold for critical alarms
        
        # Non-escalation mode: 2 attempts total (0 + 3 min, then 3 + 3 min = 6 min total)
        self.non_escalation_max_attempts = 2
        
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
        asyncio.create_task(self._alarm_timeout_monitor_task())
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
        
        # Skip if this alarm is already being tracked (avoid resetting repeat count)
        if reminder.id in self.active_alarms:
            logger.info(f"Reminder {reminder.id} already active — skipping duplicate delivery")
            return False
        
        if user_id not in self.user_connections:
            logger.warning(f"No active connection for user {user_id}, cannot deliver reminder")
            return False
        
        now = datetime.now()
        
        reminder_message = {
            "type": "reminder",
            "reminder_id": reminder.id,
            "title": reminder.title,
            "description": reminder.description,
            "priority": reminder.priority.value,
            "category": reminder.category,
            "scheduled_time": reminder.scheduled_time.isoformat(),
            "timestamp": now.isoformat(),
            "requires_acknowledgment": True,  # Frontend must get explicit confirmation
            "timeout_seconds": self.alarm_timeout_seconds,
            "escalation_enabled": reminder.escalation_enabled
        }
        
        try:
            success = await self._send_user_message(user_id, reminder_message)
            if success:
                logger.info(f"Reminder {reminder.id} delivered to user {user_id}")
                
                # Track as active alarm awaiting acknowledgment
                self.active_alarms[reminder.id] = {
                    "user_id": user_id,
                    "reminder": reminder,
                    "triggered_at": now,
                    "last_repeat_at": now,
                    "repeat_count": 0,
                    "priority": reminder.priority.value,
                    "category": reminder.category,
                    "caregiver_ids": reminder.caregiver_ids,
                    "escalation_enabled": reminder.escalation_enabled,
                    "escalation_threshold_minutes": getattr(reminder, 'escalation_threshold_minutes', self.default_escalation_threshold_minutes)
                }
                
                # Update reminder status to 'awaiting_acknowledgment'
                await self.db_service.update_reminder(
                    reminder.id, 
                    {
                        "status": "awaiting_acknowledgment", 
                        "delivered_at": now.isoformat(),
                        "alarm_triggered_at": now.isoformat()
                    }
                )
                
                mode = "escalation" if reminder.escalation_enabled else "non-escalation"
                logger.info(f"Alarm tracking started for reminder {reminder.id} ({mode} mode), timeout in {self.alarm_timeout_seconds}s")
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
                due_reminders = await self.db_service.get_due_reminders(5)

                for reminder_data in due_reminders:
                    rid = reminder_data.get("id", "")
                    if rid in self.active_alarms:
                        # If the reminder was snoozed by the user, clear it so it
                        # can be re-delivered now that its snooze period has expired.
                        if reminder_data.get("status") == "snoozed":
                            del self.active_alarms[rid]
                            logger.info(f"Snooze expired for {rid} — cleared from active_alarms for re-delivery")
                        else:
                            continue  # Still awaiting acknowledgment — skip

                    reminder = self._dict_to_reminder(reminder_data)

                    # For adaptive, non-critical reminders: skip worst response hours
                    if (
                        reminder.adaptive_scheduling_enabled
                        and reminder.priority.value != "critical"
                    ):
                        pattern = self.behavior_tracker.get_user_behavior_pattern(
                            user_id=reminder.user_id,
                            category=reminder.category,
                            days=7
                        )
                        if pattern.worst_response_hours and reminder.scheduled_time.hour in pattern.worst_response_hours:
                            # Advance by 1 hour at a time until clear of worst hours (cap at 24)
                            now = datetime.now()
                            adjusted = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                            for _ in range(24):
                                if adjusted.hour not in pattern.worst_response_hours:
                                    break
                                adjusted += timedelta(hours=1)

                            await self.db_service.update_reminder(
                                rid, {"scheduled_time": adjusted.isoformat()}
                            )
                            logger.info(
                                f"Adaptive scheduler: deferred {rid} (user={reminder.user_id}) "
                                f"from hour {reminder.scheduled_time.hour} → {adjusted.hour} "
                                f"(worst hours: {pattern.worst_response_hours})"
                            )
                            continue

                    await self.deliver_reminder(reminder)

                await asyncio.sleep(self.reminder_check_interval)

            except Exception as e:
                logger.error(f"Error in reminder scheduler task: {e}")
                await asyncio.sleep(60)
    
    async def _alarm_timeout_monitor_task(self):
        """
        Background task that monitors unacknowledged alarms.
        
        Two modes:
        
        1. escalation_enabled = False (Non-escalation mode):
           - First alarm at T=0
           - Wait 3 minutes (no response)
           - Repeat alarm at T=3 min
           - Wait 3 minutes (no response)
           - Mark as MISSED at T=6 min
           Total: 6 min, 2 attempts
        
        2. escalation_enabled = True (Escalation mode):
           - First alarm at T=0
           - Repeat every 3 minutes until escalation_threshold_minutes reached
           - After threshold (default 10 min), mark as MISSED + notify caregiver
           Total: Configurable (default 10 min), multiple attempts
        """
        logger.info("Started alarm timeout monitor task")
        
        while self.is_running:
            try:
                now = datetime.now()
                alarms_to_remove = []
                
                for reminder_id, alarm_data in list(self.active_alarms.items()):
                    triggered_at = alarm_data["triggered_at"]
                    last_repeat_at = alarm_data["last_repeat_at"]
                    repeat_count = alarm_data["repeat_count"]
                    user_id = alarm_data["user_id"]
                    reminder = alarm_data["reminder"]
                    escalation_enabled = alarm_data["escalation_enabled"]
                    escalation_threshold_minutes = alarm_data.get("escalation_threshold_minutes", self.default_escalation_threshold_minutes)
                    
                    # Calculate time since last alarm
                    time_since_last_alarm = (now - last_repeat_at).total_seconds()
                    time_since_first_trigger = (now - triggered_at).total_seconds()
                    
                    # Check if 3-minute timeout has passed since last alarm
                    if time_since_last_alarm >= self.alarm_timeout_seconds:
                        
                        if not escalation_enabled:
                            # NON-ESCALATION MODE: Max 2 attempts (at 0 min and 3 min)
                            if repeat_count < self.non_escalation_max_attempts - 1:
                                # Repeat the alarm (second attempt)
                                await self._repeat_alarm(reminder_id, alarm_data)
                            else:
                                # Max attempts reached - mark as MISSED (no caregiver notification)
                                logger.warning(f"Non-escalation alarm {reminder_id}: {self.non_escalation_max_attempts} attempts completed, marking as MISSED")
                                await self._mark_as_missed(reminder_id, alarm_data, notify_caregiver=False)
                                alarms_to_remove.append(reminder_id)
                        
                        else:
                            # ESCALATION MODE: Keep repeating until threshold reached
                            threshold_seconds = escalation_threshold_minutes * 60
                            
                            if time_since_first_trigger < threshold_seconds:
                                # Still within escalation window - repeat alarm
                                await self._repeat_alarm(reminder_id, alarm_data)
                            else:
                                # Threshold exceeded - mark as MISSED and notify caregiver
                                logger.error(
                                    f"Escalation alarm {reminder_id}: threshold of {escalation_threshold_minutes} min exceeded, "
                                    f"marking as MISSED and notifying caregiver"
                                )
                                await self._mark_as_missed(reminder_id, alarm_data, notify_caregiver=True)
                                alarms_to_remove.append(reminder_id)
                
                # Clean up completed alarms
                for reminder_id in alarms_to_remove:
                    del self.active_alarms[reminder_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in alarm timeout monitor: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _behavior_analysis_task(self):
        """Background task for updating user behavior patterns."""
        while self.is_running:
            try:
                user_ids = await self.db_service.get_active_user_ids()

                if user_ids:
                    logger.info(f"Running behavior pattern analysis for {len(user_ids)} active users...")
                    for user_id in user_ids:
                        try:
                            pattern = self.behavior_tracker.get_user_behavior_pattern(
                                user_id=user_id,
                                days=30
                            )
                            logger.info(
                                f"  [{user_id}] optimal_hour={pattern.optimal_reminder_hour}, "
                                f"trend={pattern.confusion_trend}, "
                                f"time_adj={pattern.recommended_time_adjustment_minutes}min, "
                                f"escalate={pattern.escalation_recommended}"
                            )
                        except Exception as user_err:
                            logger.warning(f"Pattern analysis failed for user {user_id}: {user_err}")
                else:
                    logger.info("Behavior analysis: no active users found")

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
    
    # ===== ALARM ACKNOWLEDGMENT & ESCALATION =====
    
    async def acknowledge_alarm(self, reminder_id: str, user_id: str, acknowledgment_method: str = "tap") -> bool:
        """
        Acknowledge an alarm - user has confirmed they saw/heard it.
        
        Args:
            reminder_id: Reminder identifier
            user_id: User who acknowledged
            acknowledgment_method: How they acknowledged (tap, verbal, button)
        
        Returns:
            True if acknowledged successfully
        """
        if reminder_id not in self.active_alarms:
            logger.warning(f"Alarm {reminder_id} not found in active alarms (already acknowledged or expired)")
            return False
        
        alarm_data = self.active_alarms[reminder_id]
        
        # Calculate response time
        triggered_at = alarm_data["triggered_at"]
        response_time_seconds = (datetime.now() - triggered_at).total_seconds()
        repeat_count = alarm_data["repeat_count"]
        
        logger.info(
            f"Alarm {reminder_id} acknowledged by user {user_id} "
            f"via {acknowledgment_method} after {response_time_seconds:.1f}s "
            f"({repeat_count} repeats)"
        )
        
        # Remove from active alarms (alarm is stopped)
        del self.active_alarms[reminder_id]
        
        # Update reminder status
        await self.db_service.update_reminder(
            reminder_id,
            {
                "status": "acknowledged",
                "acknowledged_at": datetime.now().isoformat(),
                "acknowledgment_method": acknowledgment_method,
                "response_time_seconds": response_time_seconds,
                "alarm_repeat_count": repeat_count
            }
        )
        
        # Log interaction for behavior tracking
        from src.features.reminder_system.reminder_models import InteractionType
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_category=alarm_data.get("category"),
            interaction_type=InteractionType.CONFIRMED,
            interaction_time=datetime.now(),
            response_time_seconds=response_time_seconds,
            user_response_text=f"Acknowledged via {acknowledgment_method}"
        )
        self.behavior_tracker.log_interaction(interaction)
        
        # Send confirmation to user
        await self._send_user_message(user_id, {
            "type": "alarm_acknowledged",
            "reminder_id": reminder_id,
            "message": "Alarm acknowledged - reminder stopped",
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    async def _repeat_alarm(self, reminder_id: str, alarm_data: Dict[str, Any]):
        """
        Repeat an alarm that wasn't acknowledged.
        
        Sends the alarm notification again to the user's device.
        """
        user_id = alarm_data["user_id"]
        reminder = alarm_data["reminder"]
        repeat_count = alarm_data["repeat_count"] + 1
        escalation_enabled = alarm_data["escalation_enabled"]
        
        now = datetime.now()
        time_since_first = (now - alarm_data["triggered_at"]).total_seconds() / 60  # in minutes
        
        logger.warning(
            f"⏰ REPEATING ALARM {reminder_id} (attempt #{repeat_count + 1}) - "
            f"User {user_id} has not acknowledged after {time_since_first:.1f} min"
        )
        
        # Update repeat tracking
        alarm_data["repeat_count"] = repeat_count
        alarm_data["last_repeat_at"] = now
        
        # Determine urgency based on mode and attempt count
        if escalation_enabled:
            urgency = "critical" if repeat_count >= 2 else "high"
        else:
            urgency = "high"  # Non-escalation second attempt is always high
        
        # Send repeat alarm to user
        repeat_message = {
            "type": "reminder_repeat",
            "reminder_id": reminder_id,
            "title": reminder.title,
            "description": reminder.description,
            "priority": reminder.priority.value,
            "category": reminder.category,
            "repeat_count": repeat_count,
            "total_attempts": repeat_count + 1,
            "message": f"⚠️ REMINDER (Attempt #{repeat_count + 1}): {reminder.title}",
            "urgency": urgency,
            "timestamp": now.isoformat(),
            "requires_acknowledgment": True,
            "escalation_enabled": escalation_enabled
        }
        
        await self._send_user_message(user_id, repeat_message)
        
        # Update database
        await self.db_service.update_reminder(
            reminder_id,
            {
                "alarm_repeat_count": repeat_count,
                "last_repeat_at": now.isoformat()
            }
        )
        
        logger.info(f"Alarm {reminder_id} repeated (attempt #{repeat_count + 1}), next check in {self.alarm_timeout_seconds}s")
    
    async def _mark_as_missed(self, reminder_id: str, alarm_data: Dict[str, Any], notify_caregiver: bool = False):
        """
        Mark a reminder as missed.
        
        Args:
            reminder_id: The reminder to mark as missed
            alarm_data: Alarm tracking data
            notify_caregiver: Whether to notify caregivers (True for escalation mode)
        """
        user_id = alarm_data["user_id"]
        reminder = alarm_data["reminder"]
        repeat_count = alarm_data["repeat_count"]
        priority = alarm_data["priority"]
        caregiver_ids = alarm_data["caregiver_ids"]
        escalation_enabled = alarm_data["escalation_enabled"]
        
        now = datetime.now()
        total_time = (now - alarm_data["triggered_at"]).total_seconds() / 60  # in minutes
        
        logger.error(
            f"🚨 ALARM MISSED: Reminder {reminder_id} ignored by user {user_id} "
            f"after {repeat_count + 1} attempts over {total_time:.1f} minutes. "
            f"Mode: {'escalation' if escalation_enabled else 'non-escalation'}, Priority: {priority}"
        )
        
        # Mark as MISSED in database
        await self.db_service.update_reminder(
            reminder_id,
            {
                "status": "missed",
                "missed_at": now.isoformat(),
                "alarm_repeat_count": repeat_count,
                "total_attempts": repeat_count + 1,
                "total_duration_minutes": total_time,
                "escalated_to_caregiver": notify_caregiver
            }
        )
        
        # Log as IGNORED interaction
        from src.features.reminder_system.reminder_models import InteractionType
        interaction = ReminderInteraction(
            reminder_id=reminder_id,
            user_id=user_id,
            reminder_category=alarm_data.get("category"),
            interaction_type=InteractionType.IGNORED,
            interaction_time=now,
            user_response_text=None,
            response_time_seconds=None  # No response
        )
        self.behavior_tracker.log_interaction(interaction)
        
        # NOTIFY CAREGIVERS if requested (escalation mode only)
        if notify_caregiver and caregiver_ids:
            logger.info(f"Notifying {len(caregiver_ids)} caregiver(s) for missed {priority} priority reminder")
            
            # Send alerts to all caregivers
            for caregiver_id in caregiver_ids:
                await self._notify_caregiver_of_missed_alarm(
                    caregiver_id, user_id, reminder, repeat_count
                )
            
            # Notify user that caregiver was contacted
            user_message = f"⚠️ You missed: {reminder.title}. Caregiver has been notified."
        else:
            # No caregiver notification
            user_message = f"You missed: {reminder.title}"
        
        # Notify user that alarm was missed
        await self._send_user_message(user_id, {
            "type": "alarm_missed",
            "reminder_id": reminder_id,
            "message": user_message,
            "total_attempts": repeat_count + 1,
            "caregiver_notified": notify_caregiver,
            "timestamp": now.isoformat()
        })
    
    async def _notify_caregiver_of_missed_alarm(
        self, 
        caregiver_id: str, 
        user_id: str, 
        reminder: Reminder,
        repeat_count: int
    ):
        """Send missed alarm alert to caregiver."""
        severity = "critical" if reminder.priority.value == "critical" else "high"
        
        alert_message = {
            "type": "missed_alarm_alert",
            "alert_id": f"alert_{reminder.id}_{datetime.now().timestamp()}",
            "user_id": user_id,
            "reminder_id": reminder.id,
            "severity": severity,
            "priority": reminder.priority.value,
            "category": reminder.category,
            "message": (
                f"🚨 MISSED ALARM: Patient has not responded to {reminder.title} "
                f"after {repeat_count} attempts. Please check on patient immediately."
            ),
            "reminder_title": reminder.title,
            "reminder_description": reminder.description,
            "repeat_attempts": repeat_count,
            "timestamp": datetime.now().isoformat(),
            "requires_action": True
        }
        
        # Send via WebSocket
        await self._send_caregiver_message(caregiver_id, alert_message)
        
        # Also use caregiver notifier for multi-channel notification
        self.caregiver_notifier.send_missed_reminder_alert(
            caregiver_id=caregiver_id,
            user_id=user_id,
            reminder=reminder,
            missed_count=repeat_count + 1
        )
        
        logger.info(f"Caregiver {caregiver_id} notified of missed alarm {reminder.id}")
    
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
        """Process adaptive scheduling based on interaction.

        Note: behavior_tracker.log_interaction is called inside
        scheduler.process_reminder_response — do not call it here to avoid
        recording the same interaction twice in the in-memory cache.
        """
        reminder = self._dict_to_reminder(reminder_data)
        schedule_update = self.scheduler.process_reminder_response(
            reminder=reminder,
            user_response=interaction.user_response_text or "",
            response_time_seconds=interaction.response_time_seconds
        )

        # Update reminder in database if Learning 2 decided a reschedule is needed
        if schedule_update.get("reschedule_needed"):
            await self.db_service.update_reminder(
                reminder.id,
                {
                    "scheduled_time": schedule_update["next_scheduled_time"],
                    "frequency_multiplier": schedule_update.get("frequency_multiplier", 1.0),
                    "time_shift_applied_at": datetime.now().isoformat(),
                }
            )
            logger.info(
                f"Learning 2 applied: reminder {reminder.id} rescheduled to "
                f"{schedule_update['next_scheduled_time']} "
                f"(multiplier={schedule_update.get('frequency_multiplier', 1.0):.2f})"
            )
    
    async def _send_caregiver_alerts(
        self, 
        reminder_data: Dict[str, Any], 
        interaction: ReminderInteraction,
        analysis_result: Dict[str, Any]
    ):
        """Send alerts to caregivers if needed."""
        raw_cids = reminder_data.get("caregiver_ids", [])
        caregiver_ids = raw_cids if isinstance(raw_cids, list) else json.loads(raw_cids)
        
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
            caregiver_ids=(lambda v: v if isinstance(v, list) else json.loads(v))(reminder_data.get("caregiver_ids", [])),
            adaptive_scheduling_enabled=reminder_data.get("adaptive_scheduling_enabled", True),
            escalation_enabled=reminder_data.get("escalation_enabled", True),
            status=ReminderStatus(reminder_data.get("status", "active"))
        )