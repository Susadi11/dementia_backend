"""
Adaptive Reminder Scheduler

Dynamically adjusts reminder schedules based on user behavior patterns,
cognitive decline indicators, and interaction history.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .reminder_models import (
    Reminder, ReminderInteraction, ReminderStatus, 
    InteractionType, ReminderPriority
)
from .behavior_tracker import BehaviorTracker
from .reminder_analyzer import PittBasedReminderAnalyzer

logger = logging.getLogger(__name__)


class AdaptiveReminderScheduler:
    """
    Adaptively schedules reminders based on learned user behavior.
    
    Key features:
    - Learns optimal reminder times from user responses
    - Adjusts frequency based on completion rates
    - Escalates to caregivers when needed
    - Avoids times when user consistently unresponsive
    """
    
    def __init__(
        self,
        behavior_tracker: Optional[BehaviorTracker] = None,
        analyzer: Optional[PittBasedReminderAnalyzer] = None,
        db_service=None
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            behavior_tracker: BehaviorTracker instance
            analyzer: PittBasedReminderAnalyzer instance
            db_service: Database service for persistence
        """
        self.behavior_tracker = behavior_tracker or BehaviorTracker(db_service)
        self.analyzer = analyzer or PittBasedReminderAnalyzer()
        self.db_service = db_service
    
    def process_reminder_response(
        self,
        reminder: Reminder,
        user_response: str,
        audio_path: Optional[str] = None,
        response_time_seconds: Optional[float] = None
    ) -> Dict:
        """
        Process user's response to a reminder and update accordingly.
        
        Args:
            reminder: Reminder object
            user_response: User's text response
            audio_path: Optional audio recording path
            response_time_seconds: Time taken to respond
        
        Returns:
            Dict with analysis results and actions taken
        """
        try:
            # Analyze response using Pitt-based analyzer
            reminder_context = {
                'priority': reminder.priority,
                'category': reminder.category,
                'title': reminder.title
            }
            
            analysis = self.analyzer.analyze_reminder_response(
                user_response=user_response,
                audio_path=audio_path,
                reminder_context=reminder_context
            )
            
            # Create interaction record
            interaction = ReminderInteraction(
                reminder_id=reminder.id,
                user_id=reminder.user_id,
                interaction_type=InteractionType(analysis['interaction_type']),
                interaction_time=datetime.now(),
                user_response_text=user_response,
                user_response_audio_path=audio_path,
                cognitive_risk_score=analysis['cognitive_risk_score'],
                confusion_detected=analysis['confusion_detected'],
                memory_issue_detected=analysis['memory_issue_detected'],
                uncertainty_detected=analysis['uncertainty_detected'],
                features=analysis['features'],
                recommended_action=analysis['recommended_action'],
                caregiver_alert_triggered=analysis['caregiver_alert_needed'],
                response_time_seconds=response_time_seconds
            )
            
            # Log interaction for behavior learning
            self.behavior_tracker.log_interaction(interaction)
            
            # Execute recommended action
            action_result = self._execute_action(
                reminder, interaction, analysis
            )
            
            return {
                'analysis': analysis,
                'interaction': interaction.dict(),
                'action_result': action_result,
                'reminder_updated': action_result.get('reminder_updated', False),
                'caregiver_notified': action_result.get('caregiver_notified', False)
            }
            
        except Exception as e:
            logger.error(f"Error processing reminder response: {e}", exc_info=True)
            return {
                'error': str(e),
                'analysis': {},
                'interaction': {},
                'action_result': {}
            }
    
    def get_optimal_reminder_schedule(
        self,
        reminder: Reminder,
        days_analysis: int = 30
    ) -> Dict:
        """
        Calculate optimal schedule for a reminder based on behavior patterns.
        
        Args:
            reminder: Reminder to schedule
            days_analysis: Days of history to analyze
        
        Returns:
            Dict with scheduling recommendations
        """
        try:
            # Get behavior pattern
            pattern = self.behavior_tracker.get_user_behavior_pattern(
                user_id=reminder.user_id,
                reminder_id=reminder.id,
                category=reminder.category,
                days=days_analysis
            )
            
            # Calculate optimal time
            optimal_time = self._calculate_optimal_time(reminder, pattern)
            
            # Calculate frequency adjustment
            frequency_multiplier = pattern.recommended_frequency_multiplier
            
            # Determine urgency level
            urgency = self._determine_urgency(reminder, pattern)
            
            # Check if escalation needed
            escalation_needed = pattern.escalation_recommended
            
            return {
                'optimal_time': optimal_time,
                'frequency_multiplier': frequency_multiplier,
                'urgency_level': urgency,
                'escalation_needed': escalation_needed,
                'time_adjustment_minutes': pattern.recommended_time_adjustment_minutes,
                'worst_hours': pattern.worst_response_hours,
                'pattern_stats': {
                    'total_reminders': pattern.total_reminders,
                    'confirmation_rate': pattern.confirmed_count / pattern.total_reminders if pattern.total_reminders > 0 else 0,
                    'confusion_trend': pattern.confusion_trend,
                    'avg_cognitive_risk': pattern.avg_cognitive_risk_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal schedule: {e}", exc_info=True)
            return self._default_schedule()
    
    def should_send_reminder_now(
        self,
        reminder: Reminder,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Determine if reminder should be sent at current time.
        
        Args:
            reminder: Reminder to evaluate
            current_time: Optional current time (defaults to now)
        
        Returns:
            Tuple of (should_send: bool, reason: str)
        """
        current_time = current_time or datetime.now()
        
        # Check if reminder is active
        if reminder.status != ReminderStatus.ACTIVE:
            return False, f"Reminder status is {reminder.status}"
        
        # Check if scheduled time has passed
        if current_time < reminder.scheduled_time:
            return False, "Scheduled time not reached"
        
        # Get behavior pattern
        pattern = self.behavior_tracker.get_user_behavior_pattern(
            user_id=reminder.user_id,
            reminder_id=reminder.id,
            category=reminder.category,
            days=7  # Recent behavior
        )
        
        # Check if current hour is in worst response hours
        if current_time.hour in pattern.worst_response_hours:
            # For critical reminders, send anyway
            if reminder.priority == ReminderPriority.CRITICAL:
                return True, "Critical reminder - sending despite poor response hour"
            else:
                return False, f"Hour {current_time.hour} has poor response history"
        
        # Check for recent confusion patterns
        if pattern.confusion_trend == "declining" and pattern.avg_cognitive_risk_score and pattern.avg_cognitive_risk_score > 0.7:
            # Simplify reminder or notify caregiver
            return True, "User showing cognitive decline - sending with caregiver notification"
        
        return True, "Optimal time for reminder"
    
    def reschedule_reminder(
        self,
        reminder: Reminder,
        delay_minutes: Optional[int] = None,
        reason: str = "user_requested"
    ) -> Reminder:
        """
        Reschedule a reminder based on user behavior or request.
        
        Args:
            reminder: Reminder to reschedule
            delay_minutes: Minutes to delay (uses adaptive calculation if None)
            reason: Reason for rescheduling
        
        Returns:
            Updated Reminder object
        """
        try:
            if delay_minutes is None:
                # Use adaptive calculation
                schedule_info = self.get_optimal_reminder_schedule(reminder)
                delay_minutes = schedule_info['time_adjustment_minutes'] or 15
            
            # Update scheduled time
            reminder.scheduled_time = datetime.now() + timedelta(minutes=delay_minutes)
            reminder.status = ReminderStatus.SNOOZED
            reminder.updated_at = datetime.now()
            
            logger.info(
                f"Rescheduled reminder {reminder.id} by {delay_minutes} minutes. "
                f"Reason: {reason}"
            )
            
            # Persist to database
            if self.db_service:
                self.db_service.update_reminder(reminder.dict())
            
            return reminder
            
        except Exception as e:
            logger.error(f"Error rescheduling reminder: {e}", exc_info=True)
            return reminder
    
    def _execute_action(
        self,
        reminder: Reminder,
        interaction: ReminderInteraction,
        analysis: Dict
    ) -> Dict:
        """Execute recommended action based on analysis."""
        action = analysis['recommended_action']
        result = {'action': action}
        
        try:
            if action == 'mark_completed':
                reminder.status = ReminderStatus.COMPLETED
                reminder.completed_at = datetime.now()
                result['reminder_updated'] = True
                
            elif action == 'snooze_reminder':
                self.reschedule_reminder(reminder, delay_minutes=15, reason="user_delayed")
                result['reminder_updated'] = True
                result['rescheduled_time'] = reminder.scheduled_time
                
            elif action in ['escalate_to_caregiver_urgent', 'simplify_reminder_with_context']:
                if analysis['caregiver_alert_needed'] and reminder.caregiver_ids:
                    result['caregiver_notified'] = self._notify_caregivers(
                        reminder, interaction, urgent=(action == 'escalate_to_caregiver_urgent')
                    )
                
            elif action == 'provide_context_and_repeat':
                # Reschedule with shorter delay
                self.reschedule_reminder(reminder, delay_minutes=5, reason="memory_issue_detected")
                result['reminder_updated'] = True
                
            elif action == 'increase_monitoring':
                # Adjust frequency
                if reminder.adaptive_scheduling_enabled:
                    result['frequency_increased'] = True
            
            # Persist reminder updates
            if result.get('reminder_updated') and self.db_service:
                self.db_service.update_reminder(reminder.dict())
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}", exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def _notify_caregivers(
        self,
        reminder: Reminder,
        interaction: ReminderInteraction,
        urgent: bool = False
    ) -> bool:
        """Send notification to caregivers."""
        try:
            from .caregiver_notifier import CaregiverNotifier
            
            notifier = CaregiverNotifier(self.db_service)
            
            alert_type = "missed_critical" if urgent else "confusion_detected"
            severity = "critical" if urgent else "high"
            
            message = self._generate_caregiver_message(reminder, interaction, urgent)
            
            for caregiver_id in reminder.caregiver_ids:
                notifier.send_alert(
                    caregiver_id=caregiver_id,
                    user_id=reminder.user_id,
                    reminder=reminder,
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    interaction=interaction
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error notifying caregivers: {e}", exc_info=True)
            return False
    
    def _generate_caregiver_message(
        self,
        reminder: Reminder,
        interaction: ReminderInteraction,
        urgent: bool
    ) -> str:
        """Generate message for caregiver alert."""
        if urgent:
            return (
                f"URGENT: Patient {reminder.user_id} showing high cognitive risk "
                f"(score: {interaction.cognitive_risk_score:.2f}) for critical reminder: "
                f"{reminder.title}. Immediate attention needed."
            )
        elif interaction.confusion_detected:
            return (
                f"Patient {reminder.user_id} appears confused about reminder: "
                f"{reminder.title}. Response: \"{interaction.user_response_text}\". "
                f"May need assistance."
            )
        elif interaction.memory_issue_detected:
            return (
                f"Patient {reminder.user_id} showing memory issues with reminder: "
                f"{reminder.title}. Consider in-person reminder or verification."
            )
        else:
            return (
                f"Patient {reminder.user_id} needs attention for reminder: "
                f"{reminder.title}. Cognitive risk score: {interaction.cognitive_risk_score:.2f}"
            )
    
    def _calculate_optimal_time(
        self,
        reminder: Reminder,
        pattern
    ) -> datetime:
        """Calculate optimal time for reminder."""
        base_time = reminder.scheduled_time
        
        # Adjust based on optimal hour if available
        if pattern.optimal_reminder_hour is not None:
            base_time = base_time.replace(hour=pattern.optimal_reminder_hour)
        
        # Apply time adjustment
        if pattern.recommended_time_adjustment_minutes:
            base_time += timedelta(minutes=pattern.recommended_time_adjustment_minutes)
        
        # Avoid worst hours
        while base_time.hour in pattern.worst_response_hours:
            base_time += timedelta(hours=1)
        
        return base_time
    
    def _determine_urgency(self, reminder: Reminder, pattern) -> str:
        """Determine urgency level based on reminder and pattern."""
        if reminder.priority == ReminderPriority.CRITICAL:
            return "critical"
        
        if pattern.escalation_recommended:
            return "high"
        
        if pattern.confusion_trend == "declining":
            return "high"
        
        if pattern.avg_cognitive_risk_score and pattern.avg_cognitive_risk_score > 0.6:
            return "medium"
        
        return "normal"
    
    def _default_schedule(self) -> Dict:
        """Return default schedule when analysis fails."""
        return {
            'optimal_time': datetime.now() + timedelta(hours=1),
            'frequency_multiplier': 1.0,
            'urgency_level': 'normal',
            'escalation_needed': False,
            'time_adjustment_minutes': 0,
            'worst_hours': [],
            'pattern_stats': {}
        }
