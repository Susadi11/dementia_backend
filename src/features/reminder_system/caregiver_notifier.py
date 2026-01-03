"""
Caregiver Notifier

Handles sending alerts and notifications to caregivers when:
- Critical reminders are missed
- User shows confusion or cognitive decline
- Escalation protocols are triggered
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from .reminder_models import (
    Reminder, ReminderInteraction, CaregiverAlert
)

logger = logging.getLogger(__name__)


class CaregiverNotifier:
    """
    Manages caregiver notifications and escalation protocols.
    
    Handles multiple notification channels:
    - Push notifications (mobile app)
    - SMS alerts
    - Email notifications
    - In-app messages
    """
    
    def __init__(self, db_service=None, notification_service=None):
        """
        Initialize caregiver notifier.
        
        Args:
            db_service: Database service for persistence
            notification_service: External notification service (SMS, push, email)
        """
        self.db_service = db_service
        self.notification_service = notification_service
        self.alert_cache: List[CaregiverAlert] = []
    
    def send_alert(
        self,
        caregiver_id: str,
        user_id: str,
        reminder: Reminder,
        alert_type: str,
        severity: str,
        message: str,
        interaction: Optional[ReminderInteraction] = None,
        missed_count: int = 0
    ) -> bool:
        """
        Send alert to caregiver.
        
        Args:
            caregiver_id: Caregiver identifier
            user_id: Patient user identifier
            reminder: Reminder that triggered alert
            alert_type: Type of alert (missed_critical, confusion_detected, etc.)
            severity: Severity level (low, medium, high, critical)
            message: Alert message
            interaction: Optional interaction that triggered alert
            missed_count: Number of times reminder was missed
        
        Returns:
            True if alert sent successfully
        """
        try:
            # Create alert record
            alert = CaregiverAlert(
                caregiver_id=caregiver_id,
                user_id=user_id,
                reminder_id=reminder.id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                reminder_title=reminder.title,
                missed_count=missed_count,
                cognitive_risk_score=interaction.cognitive_risk_score if interaction else None,
                user_response=interaction.user_response_text if interaction else None,
                created_at=datetime.now()
            )
            
            # Cache alert
            self.alert_cache.append(alert)
            
            # Persist to database
            if self.db_service:
                self.db_service.save_caregiver_alert(alert.dict())
            
            # Send through notification channels
            notification_sent = self._send_notification(alert)
            
            logger.info(
                f"Alert sent to caregiver {caregiver_id}: "
                f"type={alert_type}, severity={severity}, "
                f"user={user_id}, reminder={reminder.title}"
            )
            
            return notification_sent
            
        except Exception as e:
            logger.error(f"Error sending caregiver alert: {e}", exc_info=True)
            return False
    
    def send_missed_reminder_alert(
        self,
        caregiver_id: str,
        user_id: str,
        reminder: Reminder,
        missed_count: int
    ) -> bool:
        """
        Send alert when critical reminder is missed.
        
        Args:
            caregiver_id: Caregiver identifier
            user_id: Patient identifier
            reminder: Missed reminder
            missed_count: Number of times missed
        
        Returns:
            True if alert sent successfully
        """
        severity = "critical" if missed_count >= 3 else "high"
        
        message = (
            f"Patient has missed {missed_count} reminder(s) for: {reminder.title}. "
            f"Last scheduled: {reminder.scheduled_time.strftime('%H:%M on %b %d')}. "
        )
        
        if reminder.category == "medication":
            message += "This is a medication reminder - please verify patient status immediately."
        elif reminder.category == "appointment":
            message += "This is a medical appointment reminder - please check with patient."
        else:
            message += "Please check on patient and provide assistance if needed."
        
        return self.send_alert(
            caregiver_id=caregiver_id,
            user_id=user_id,
            reminder=reminder,
            alert_type="missed_critical",
            severity=severity,
            message=message,
            missed_count=missed_count
        )
    
    def send_confusion_alert(
        self,
        caregiver_id: str,
        user_id: str,
        reminder: Reminder,
        interaction: ReminderInteraction
    ) -> bool:
        """
        Send alert when user shows confusion.
        
        Args:
            caregiver_id: Caregiver identifier
            user_id: Patient identifier
            reminder: Reminder that triggered confusion
            interaction: User interaction details
        
        Returns:
            True if alert sent successfully
        """
        message = (
            f"Patient appears confused about reminder: {reminder.title}. "
            f"Response: \"{interaction.user_response_text}\". "
            f"Cognitive risk score: {interaction.cognitive_risk_score:.2f}. "
            f"Patient may need in-person assistance or clarification."
        )
        
        return self.send_alert(
            caregiver_id=caregiver_id,
            user_id=user_id,
            reminder=reminder,
            alert_type="confusion_detected",
            severity="high",
            message=message,
            interaction=interaction
        )
    
    def send_cognitive_decline_alert(
        self,
        caregiver_id: str,
        user_id: str,
        avg_risk_score: float,
        trend: str,
        recent_interactions: List[ReminderInteraction]
    ) -> bool:
        """
        Send alert when cognitive decline pattern detected.
        
        Args:
            caregiver_id: Caregiver identifier
            user_id: Patient identifier
            avg_risk_score: Average cognitive risk score
            trend: Trend direction (declining, stable, improving)
            recent_interactions: Recent interactions showing pattern
        
        Returns:
            True if alert sent successfully
        """
        message = (
            f"Cognitive decline pattern detected for patient {user_id}. "
            f"Average risk score: {avg_risk_score:.2f}. "
            f"Trend: {trend}. "
            f"Based on {len(recent_interactions)} recent interactions. "
            f"Consider scheduling medical evaluation."
        )
        
        # Create a synthetic reminder for the alert
        from .reminder_models import Reminder, ReminderPriority
        
        synthetic_reminder = Reminder(
            id="cognitive_decline_alert",
            user_id=user_id,
            title="Cognitive Decline Pattern Detected",
            priority=ReminderPriority.CRITICAL,
            category="health_monitoring",
            scheduled_time=datetime.now(),
            caregiver_ids=[caregiver_id]
        )
        
        return self.send_alert(
            caregiver_id=caregiver_id,
            user_id=user_id,
            reminder=synthetic_reminder,
            alert_type="cognitive_decline",
            severity="critical",
            message=message
        )
    
    def create_confusion_alert(
        self,
        reminder: Reminder,
        interaction: ReminderInteraction
    ) -> CaregiverAlert:
        """
        Create a confusion alert for caregivers.
        
        Args:
            reminder: Reminder that triggered confusion
            interaction: User interaction details
        
        Returns:
            CaregiverAlert object
        """
        return CaregiverAlert(
            caregiver_id=reminder.caregiver_ids[0] if reminder.caregiver_ids else "unknown",
            user_id=reminder.user_id,
            reminder_id=reminder.id,
            alert_type="confusion_detected",
            severity="high",
            message=(
                f"Patient appears confused about reminder: {reminder.title}. "
                f"Response: \"{interaction.user_response_text}\". "
                f"Cognitive risk score: {interaction.cognitive_risk_score:.2f}. "
                f"Patient may need in-person assistance or clarification."
            ),
            reminder_title=reminder.title,
            cognitive_risk_score=interaction.cognitive_risk_score,
            user_response=interaction.user_response_text,
            created_at=datetime.now()
        )
    
    def create_missed_reminder_alert(
        self,
        reminder: Reminder,
        missed_count: int = 1
    ) -> CaregiverAlert:
        """
        Create a missed reminder alert.
        
        Args:
            reminder: Missed reminder
            missed_count: Number of times missed
        
        Returns:
            CaregiverAlert object
        """
        severity = "critical" if missed_count >= 3 else "high"
        
        message = (
            f"Patient has missed {missed_count} reminder(s) for: {reminder.title}. "
            f"Last scheduled: {reminder.scheduled_time.strftime('%H:%M on %b %d')}. "
        )
        
        if reminder.category == "medication":
            message += "This is a medication reminder - please verify patient status immediately."
        elif reminder.category == "appointment":
            message += "This is a medical appointment reminder - please check with patient."
        else:
            message += "Please check on patient and provide assistance if needed."
        
        return CaregiverAlert(
            caregiver_id=reminder.caregiver_ids[0] if reminder.caregiver_ids else "unknown",
            user_id=reminder.user_id,
            reminder_id=reminder.id,
            alert_type="missed_reminder",
            severity=severity,
            message=message,
            reminder_title=reminder.title,
            missed_count=missed_count,
            created_at=datetime.now()
        )
    
    def create_high_risk_pattern_alert(
        self,
        user_id: str,
        reminder_id: str,
        risk_score: float,
        pattern_details: str,
        caregiver_ids: List[str]
    ) -> CaregiverAlert:
        """
        Create a high-risk pattern alert.
        
        Args:
            user_id: Patient identifier
            reminder_id: Related reminder ID
            risk_score: Cognitive risk score
            pattern_details: Description of the pattern
            caregiver_ids: List of caregiver IDs
        
        Returns:
            CaregiverAlert object
        """
        return CaregiverAlert(
            caregiver_id=caregiver_ids[0] if caregiver_ids else "unknown",
            user_id=user_id,
            reminder_id=reminder_id,
            alert_type="high_risk_pattern",
            severity="critical",
            message=(
                f"High cognitive risk pattern detected. "
                f"Risk score: {risk_score:.2f}. "
                f"Pattern: {pattern_details}. "
                f"Consider scheduling medical evaluation."
            ),
            reminder_title="Cognitive Risk Pattern Alert",
            cognitive_risk_score=risk_score,
            created_at=datetime.now(),
            context={"risk_score": risk_score, "pattern_details": pattern_details}
        )
    
    def send_daily_summary(
        self,
        caregiver_id: str,
        user_id: str,
        summary_data: Dict
    ) -> bool:
        """
        Send daily summary of patient's reminder interactions.
        
        Args:
            caregiver_id: Caregiver identifier
            user_id: Patient identifier
            summary_data: Summary statistics and highlights
        
        Returns:
            True if summary sent successfully
        """
        try:
            message = self._format_daily_summary(user_id, summary_data)
            
            # Send through email or app notification
            if self.notification_service:
                self.notification_service.send_email(
                    recipient_id=caregiver_id,
                    subject=f"Daily Reminder Summary for Patient {user_id}",
                    body=message
                )
            
            logger.info(f"Daily summary sent to caregiver {caregiver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}", exc_info=True)
            return False
    
    def acknowledge_alert(self, alert_id: str, caregiver_id: str) -> bool:
        """
        Mark alert as acknowledged by caregiver.
        
        Args:
            alert_id: Alert identifier
            caregiver_id: Caregiver who acknowledged
        
        Returns:
            True if successfully acknowledged
        """
        try:
            if self.db_service:
                self.db_service.update_alert_status(
                    alert_id=alert_id,
                    acknowledged_at=datetime.now(),
                    is_acknowledged=True
                )
            
            # Update cache
            for alert in self.alert_cache:
                if alert.id == alert_id:
                    alert.is_acknowledged = True
                    alert.acknowledged_at = datetime.now()
            
            logger.info(f"Alert {alert_id} acknowledged by {caregiver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}", exc_info=True)
            return False
    
    def resolve_alert(self, alert_id: str, caregiver_id: str) -> bool:
        """
        Mark alert as resolved.
        
        Args:
            alert_id: Alert identifier
            caregiver_id: Caregiver who resolved
        
        Returns:
            True if successfully resolved
        """
        try:
            if self.db_service:
                self.db_service.update_alert_status(
                    alert_id=alert_id,
                    resolved_at=datetime.now(),
                    is_resolved=True
                )
            
            # Update cache
            for alert in self.alert_cache:
                if alert.id == alert_id:
                    alert.is_resolved = True
                    alert.resolved_at = datetime.now()
            
            logger.info(f"Alert {alert_id} resolved by {caregiver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}", exc_info=True)
            return False
    
    def get_active_alerts(self, caregiver_id: str) -> List[CaregiverAlert]:
        """
        Get all active (unresolved) alerts for a caregiver.
        
        Args:
            caregiver_id: Caregiver identifier
        
        Returns:
            List of active alerts
        """
        try:
            if self.db_service:
                return self.db_service.get_active_alerts(caregiver_id)
            
            # Fall back to cache
            return [
                alert for alert in self.alert_cache
                if alert.caregiver_id == caregiver_id and not alert.is_resolved
            ]
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}", exc_info=True)
            return []
    
    def _send_notification(self, alert: CaregiverAlert) -> bool:
        """Send notification through configured channels."""
        try:
            if not self.notification_service:
                logger.warning("No notification service configured")
                return False
            
            # Send based on severity
            if alert.severity in ["critical", "high"]:
                # Send push notification + SMS for urgent alerts
                self.notification_service.send_push_notification(
                    recipient_id=alert.caregiver_id,
                    title=f"{alert.severity.upper()}: {alert.alert_type}",
                    body=alert.message,
                    priority="high"
                )
                
                self.notification_service.send_sms(
                    recipient_id=alert.caregiver_id,
                    message=f"{alert.alert_type}: {alert.message[:160]}"
                )
            else:
                # Send in-app notification for lower priority
                self.notification_service.send_in_app_notification(
                    recipient_id=alert.caregiver_id,
                    title=alert.alert_type,
                    body=alert.message
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}", exc_info=True)
            return False
    
    def _format_daily_summary(self, user_id: str, summary_data: Dict) -> str:
        """Format daily summary message."""
        message = f"Daily Reminder Summary for Patient {user_id}\n"
        message += f"Date: {datetime.now().strftime('%B %d, %Y')}\n\n"
        
        message += f"Total Reminders: {summary_data.get('total_reminders', 0)}\n"
        message += f"Completed: {summary_data.get('completed', 0)}\n"
        message += f"Missed: {summary_data.get('missed', 0)}\n"
        message += f"Delayed: {summary_data.get('delayed', 0)}\n\n"
        
        if summary_data.get('avg_cognitive_risk'):
            message += f"Average Cognitive Risk: {summary_data['avg_cognitive_risk']:.2f}\n"
        
        if summary_data.get('confusion_count', 0) > 0:
            message += f"⚠️ Confusion detected {summary_data['confusion_count']} time(s)\n"
        
        if summary_data.get('highlights'):
            message += "\nHighlights:\n"
            for highlight in summary_data['highlights']:
                message += f"- {highlight}\n"
        
        return message
