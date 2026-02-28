"""
Behavior Tracker

Tracks user interactions with reminders to identify patterns,
cognitive decline trends, and optimal reminder timings.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging

from .reminder_models import (
    ReminderInteraction, InteractionType, BehaviorPattern
)

logger = logging.getLogger(__name__)


class BehaviorTracker:
    """
    Tracks and analyzes user behavior patterns with reminders.
    
    Monitors:
    - Response times to reminders
    - Interaction types (confirmed, ignored, confused, etc.)
    - Cognitive risk trends
    - Time-of-day patterns
    - Category-specific behaviors
    """
    
    def __init__(self, db_service=None):
        """
        Initialize behavior tracker.
        
        Args:
            db_service: Database service for persistent storage
        """
        self.db_service = db_service
        self.interaction_cache: Dict[str, List[ReminderInteraction]] = defaultdict(list)
    
    def log_interaction(self, interaction: ReminderInteraction):
        """
        Log a user interaction with a reminder.
        
        Args:
            interaction: ReminderInteraction object with details
        """
        try:
            # Cache interaction
            cache_key = f"{interaction.user_id}_{interaction.reminder_id}"
            self.interaction_cache[cache_key].append(interaction)
            
            # Persist to database if available
            if self.db_service:
                self.db_service.save_reminder_interaction(interaction.dict())
            
            logger.info(
                f"Logged interaction: user={interaction.user_id}, "
                f"type={interaction.interaction_type}, "
                f"risk={interaction.cognitive_risk_score}"
            )
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}", exc_info=True)
    
    def get_user_behavior_pattern(
        self,
        user_id: str,
        reminder_id: Optional[str] = None,
        category: Optional[str] = None,
        days: int = 30
    ) -> BehaviorPattern:
        """
        Analyze user behavior patterns over specified period.
        
        Args:
            user_id: User identifier
            reminder_id: Optional specific reminder to analyze
            category: Optional reminder category to analyze
            days: Number of days to analyze (default 30)
        
        Returns:
            BehaviorPattern with statistics and recommendations
        """
        try:
            # Get interactions from database or cache
            interactions = self._get_interactions(
                user_id, reminder_id, category, days
            )
            
            if not interactions:
                return self._default_pattern(user_id, reminder_id, category)
            
            # Calculate statistics
            total = len(interactions)
            confirmed = sum(1 for i in interactions if i.interaction_type == InteractionType.CONFIRMED)
            ignored = sum(1 for i in interactions if i.interaction_type == InteractionType.IGNORED)
            delayed = sum(1 for i in interactions if i.interaction_type == InteractionType.DELAYED)
            confused = sum(1 for i in interactions if i.interaction_type == InteractionType.CONFUSED)
            
            # Response times
            response_times = [
                i.response_time_seconds for i in interactions 
                if i.response_time_seconds is not None
            ]
            avg_response_time = statistics.mean(response_times) if response_times else None
            
            # Cognitive risk trend
            risk_scores = [
                i.cognitive_risk_score for i in interactions 
                if i.cognitive_risk_score is not None
            ]
            avg_risk = statistics.mean(risk_scores) if risk_scores else None
            
            # Time-of-day analysis
            optimal_hour = self._find_optimal_hour(interactions)
            worst_hours = self._find_worst_hours(interactions)
            
            # Trend analysis
            confusion_trend = self._analyze_confusion_trend(interactions)
            
            # Generate recommendations
            freq_multiplier = self._calculate_frequency_multiplier(
                confirmed, ignored, confused, total
            )
            time_adjustment = self._calculate_time_adjustment(interactions)
            escalation_needed = self._should_escalate(
                confirmed, ignored, confused, avg_risk, total
            )
            
            return BehaviorPattern(
                user_id=user_id,
                reminder_id=reminder_id,
                category=category,
                total_reminders=total,
                confirmed_count=confirmed,
                ignored_count=ignored,
                delayed_count=delayed,
                confused_count=confused,
                avg_response_time_seconds=avg_response_time,
                optimal_reminder_hour=optimal_hour,
                worst_response_hours=worst_hours,
                avg_cognitive_risk_score=avg_risk,
                confusion_trend=confusion_trend,
                recommended_frequency_multiplier=freq_multiplier,
                recommended_time_adjustment_minutes=time_adjustment,
                escalation_recommended=escalation_needed,
                last_updated=datetime.now(),
                analysis_period_days=days
            )
            
        except Exception as e:
            logger.error(f"Error analyzing behavior pattern: {e}", exc_info=True)
            return self._default_pattern(user_id, reminder_id, category)
    
    def _get_interactions(
        self,
        user_id: str,
        reminder_id: Optional[str],
        category: Optional[str],
        days: int
    ) -> List[ReminderInteraction]:
        """Get interactions from database or cache."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Try database first
        if self.db_service:
            try:
                filters = {'user_id': user_id}
                if reminder_id:
                    filters['reminder_id'] = reminder_id
                if category:
                    filters['category'] = category
                
                interactions = self.db_service.get_reminder_interactions(
                    filters=filters,
                    start_date=cutoff_date
                )
                return interactions
            except Exception as e:
                logger.warning(f"Database query failed: {e}")
        
        # Fall back to cache
        all_interactions = []
        for key, interactions in self.interaction_cache.items():
            if user_id in key:
                filtered = [
                    i for i in interactions
                    if i.interaction_time >= cutoff_date
                ]
                if reminder_id:
                    filtered = [i for i in filtered if i.reminder_id == reminder_id]
                all_interactions.extend(filtered)
        
        return all_interactions
    
    def _find_optimal_hour(self, interactions: List[ReminderInteraction]) -> Optional[int]:
        """Find the hour with best response rate."""
        hour_stats = defaultdict(lambda: {'confirmed': 0, 'total': 0})
        
        for interaction in interactions:
            hour = interaction.interaction_time.hour
            hour_stats[hour]['total'] += 1
            if interaction.interaction_type == InteractionType.CONFIRMED:
                hour_stats[hour]['confirmed'] += 1
        
        if not hour_stats:
            return None
        
        # Find hour with highest confirmation rate
        best_hour = max(
            hour_stats.items(),
            key=lambda x: x[1]['confirmed'] / x[1]['total'] if x[1]['total'] > 0 else 0
        )
        
        return best_hour[0] if best_hour[1]['total'] >= 3 else None
    
    def _find_worst_hours(self, interactions: List[ReminderInteraction]) -> List[int]:
        """Find hours with worst response rates."""
        hour_stats = defaultdict(lambda: {'confused': 0, 'ignored': 0, 'total': 0})
        
        for interaction in interactions:
            hour = interaction.interaction_time.hour
            hour_stats[hour]['total'] += 1
            if interaction.interaction_type in [InteractionType.CONFUSED, InteractionType.IGNORED]:
                hour_stats[hour]['confused'] += 1
        
        # Find hours with >50% poor responses
        worst_hours = [
            hour for hour, stats in hour_stats.items()
            if stats['total'] >= 3 and stats['confused'] / stats['total'] > 0.5
        ]
        
        return sorted(worst_hours)
    
    def _analyze_confusion_trend(self, interactions: List[ReminderInteraction]) -> str:
        """Analyze if confusion is improving, stable, or declining."""
        if len(interactions) < 10:
            return "insufficient_data"
        
        # Split into recent and older interactions
        mid_point = len(interactions) // 2
        older = interactions[:mid_point]
        recent = interactions[mid_point:]
        
        older_confusion_rate = sum(
            1 for i in older if i.interaction_type == InteractionType.CONFUSED
        ) / len(older)
        
        recent_confusion_rate = sum(
            1 for i in recent if i.interaction_type == InteractionType.CONFUSED
        ) / len(recent)
        
        # Compare rates
        if recent_confusion_rate < older_confusion_rate - 0.1:
            return "improving"
        elif recent_confusion_rate > older_confusion_rate + 0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_frequency_multiplier(
        self,
        confirmed: int,
        ignored: int,
        confused: int,
        total: int
    ) -> float:
        """Calculate recommended frequency adjustment."""
        if total < 5:
            return 1.0
        
        confirmation_rate = confirmed / total
        problem_rate = (ignored + confused) / total
        
        # Low confirmation rate - increase frequency
        if confirmation_rate < 0.3:
            return 1.5
        
        # High problem rate - increase frequency
        if problem_rate > 0.5:
            return 1.3
        
        # Good performance - maintain or slightly decrease
        if confirmation_rate > 0.7:
            return 0.9
        
        return 1.0
    
    def _calculate_time_adjustment(
        self,
        interactions: List[ReminderInteraction]
    ) -> int:
        """Calculate recommended time adjustment in minutes."""
        if len(interactions) < 5:
            return 0
        
        # Analyze response patterns by time
        ignored_times = [
            i.interaction_time.hour for i in interactions
            if i.interaction_type == InteractionType.IGNORED
        ]
        
        if not ignored_times:
            return 0
        
        # If consistently ignored at certain hour, suggest shift
        most_ignored_hour = max(set(ignored_times), key=ignored_times.count)
        ignored_count = ignored_times.count(most_ignored_hour)
        
        if ignored_count >= 3:
            # Suggest shifting by 30-60 minutes
            return 30
        
        return 0
    
    def _should_escalate(
        self,
        confirmed: int,
        ignored: int,
        confused: int,
        avg_risk: Optional[float],
        total: int
    ) -> bool:
        """Determine if situation should be escalated to caregiver."""
        if total < 5:
            return False
        
        # High ignore rate
        if ignored / total > 0.6:
            return True
        
        # High confusion rate (more than 1/3 confused responses)
        if confused / total > 0.33:
            return True
        
        # High cognitive risk
        if avg_risk and avg_risk > 0.7:
            return True
        
        # Very low confirmation rate
        if confirmed / total < 0.2:
            return True
        
        return False
    
    def _default_pattern(
        self,
        user_id: str,
        reminder_id: Optional[str],
        category: Optional[str]
    ) -> BehaviorPattern:
        """Return default pattern when no data available."""
        return BehaviorPattern(
            user_id=user_id,
            reminder_id=reminder_id,
            category=category,
            total_reminders=0,
            confirmed_count=0,
            ignored_count=0,
            delayed_count=0,
            confused_count=0,
            confusion_trend="insufficient_data",
            last_updated=datetime.now()
        )
