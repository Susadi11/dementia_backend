"""
Weekly Report Generator for Risk Monitoring

Generates comprehensive weekly reports for dementia patients including:
- Cognitive risk trends
- Reminder completion statistics
- Confusion patterns
- Caregiver alert frequency
- Recommendations for intervention
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging
from pydantic import BaseModel

from .reminder_models import ReminderInteraction, InteractionType, CaregiverAlert
from .behavior_tracker import BehaviorTracker

logger = logging.getLogger(__name__)


class DailyRiskSummary(BaseModel):
    """Daily summary of risk metrics."""
    date: str
    avg_cognitive_risk: float
    confusion_count: int
    total_interactions: int
    completion_rate: float
    alert_count: int


class WeeklyCognitiveReport(BaseModel):
    """Comprehensive weekly cognitive risk report."""
    user_id: str
    report_period_start: str
    report_period_end: str
    generated_at: str
    
    # Overall Statistics
    total_reminders: int
    completed_reminders: int
    missed_reminders: int
    completion_rate: float
    
    # Cognitive Health Metrics
    avg_cognitive_risk_score: float
    peak_cognitive_risk_score: float
    lowest_cognitive_risk_score: float
    risk_trend: str  # "improving", "stable", "declining"
    
    # Interaction Patterns
    confusion_count: int
    memory_issue_count: int
    confirmed_count: int
    ignored_count: int
    delayed_count: int
    
    # Daily Breakdown
    daily_summaries: List[DailyRiskSummary]
    
    # Caregiver Alerts
    total_alerts: int
    critical_alerts: int
    high_priority_alerts: int
    unresolved_alerts: int
    
    # Time Analysis
    best_response_hours: List[int]
    worst_response_hours: List[int]
    avg_response_time_seconds: float
    
    # Category Performance
    category_breakdown: Dict[str, Dict[str, Any]]
    
    # Recommendations
    risk_level: str  # "low", "moderate", "high", "critical"
    recommendations: List[str]
    intervention_needed: bool
    escalation_required: bool
    
    # Week-over-Week Comparison
    previous_week_avg_risk: Optional[float] = None
    risk_change_percentage: Optional[float] = None


class WeeklyReportGenerator:
    """
    Generates weekly risk monitoring reports for dementia patients.
    """
    
    def __init__(self, behavior_tracker: BehaviorTracker, db_service=None):
        """
        Initialize weekly report generator.
        
        Args:
            behavior_tracker: BehaviorTracker instance
            db_service: Database service for data retrieval
        """
        self.behavior_tracker = behavior_tracker
        self.db_service = db_service
    
    def generate_weekly_report(
        self,
        user_id: str,
        end_date: Optional[datetime] = None,
        caregiver_ids: Optional[List[str]] = None
    ) -> WeeklyCognitiveReport:
        """
        Generate comprehensive weekly report for a user.
        
        Args:
            user_id: User identifier
            end_date: End date for report (default: today)
            caregiver_ids: Optional list of caregiver IDs (for compatibility)
        
        Returns:
            WeeklyCognitiveReport with all metrics and analysis
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Generating weekly report for {user_id} ({start_date.date()} to {end_date.date()})")
        
        # Get all interactions for the week
        interactions = self._get_interactions(user_id, start_date, end_date)
        
        if not interactions:
            return self._empty_report(user_id, start_date, end_date)
        
        # Get caregiver alerts
        alerts = self._get_alerts(user_id, start_date, end_date)
        
        # Calculate statistics
        stats = self._calculate_statistics(interactions, alerts)
        
        # Generate daily summaries
        daily_summaries = self._generate_daily_summaries(interactions, alerts, start_date, end_date)
        
        # Analyze trends
        risk_trend = self._analyze_risk_trend(daily_summaries)
        
        # Get previous week for comparison
        previous_week_stats = self._get_previous_week_comparison(user_id, start_date)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stats, risk_trend, alerts)
        
        # Determine risk level and intervention needs
        risk_level = self._determine_risk_level(stats['avg_cognitive_risk'])
        intervention_needed = self._needs_intervention(stats, risk_trend, alerts)
        escalation_required = self._needs_escalation(stats, alerts)
        
        # Build report
        report = WeeklyCognitiveReport(
            user_id=user_id,
            report_period_start=start_date.isoformat(),
            report_period_end=end_date.isoformat(),
            generated_at=datetime.now().isoformat(),
            
            # Overall Statistics
            total_reminders=stats['total_reminders'],
            completed_reminders=stats['completed_reminders'],
            missed_reminders=stats['missed_reminders'],
            completion_rate=stats['completion_rate'],
            
            # Cognitive Health
            avg_cognitive_risk_score=stats['avg_cognitive_risk'],
            peak_cognitive_risk_score=stats['peak_risk'],
            lowest_cognitive_risk_score=stats['lowest_risk'],
            risk_trend=risk_trend,
            
            # Interaction Patterns
            confusion_count=stats['confusion_count'],
            memory_issue_count=stats['memory_issue_count'],
            confirmed_count=stats['confirmed_count'],
            ignored_count=stats['ignored_count'],
            delayed_count=stats['delayed_count'],
            
            # Daily Breakdown
            daily_summaries=daily_summaries,
            
            # Alerts
            total_alerts=stats['total_alerts'],
            critical_alerts=stats['critical_alerts'],
            high_priority_alerts=stats['high_priority_alerts'],
            unresolved_alerts=stats['unresolved_alerts'],
            
            # Time Analysis
            best_response_hours=stats['best_hours'],
            worst_response_hours=stats['worst_hours'],
            avg_response_time_seconds=stats['avg_response_time'],
            
            # Category Performance
            category_breakdown=stats['category_breakdown'],
            
            # Recommendations
            risk_level=risk_level,
            recommendations=recommendations,
            intervention_needed=intervention_needed,
            escalation_required=escalation_required,
            
            # Week-over-Week
            previous_week_avg_risk=previous_week_stats.get('avg_risk'),
            risk_change_percentage=previous_week_stats.get('change_percentage')
        )
        
        # Save report to database
        if self.db_service:
            self.db_service.save_weekly_report(report.dict())
        
        logger.info(f"Weekly report generated: risk_level={risk_level}, intervention={intervention_needed}")
        
        return report
    
    def _get_interactions(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ReminderInteraction]:
        """Retrieve interactions for date range."""
        try:
            if self.db_service:
                return self.db_service.get_interactions_in_range(
                    user_id, start_date, end_date
                )
            else:
                # Use cached data
                cache_key = f"{user_id}_*"
                all_interactions = []
                for key, interactions in self.behavior_tracker.interaction_cache.items():
                    if key.startswith(user_id):
                        all_interactions.extend([
                            i for i in interactions
                            if start_date <= i.interaction_time <= end_date
                        ])
                return all_interactions
        except Exception as e:
            logger.error(f"Error retrieving interactions: {e}")
            return []
    
    def _get_alerts(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CaregiverAlert]:
        """Retrieve caregiver alerts for date range."""
        try:
            if self.db_service:
                return self.db_service.get_alerts_in_range(
                    user_id, start_date, end_date
                )
            return []
        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []
    
    def _calculate_statistics(
        self,
        interactions: List[ReminderInteraction],
        alerts: List[CaregiverAlert]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics."""
        total = len(interactions)
        
        if total == 0:
            return self._default_statistics()
        
        # Count interaction types
        confirmed = sum(1 for i in interactions if i.interaction_type == InteractionType.CONFIRMED)
        ignored = sum(1 for i in interactions if i.interaction_type == InteractionType.IGNORED)
        delayed = sum(1 for i in interactions if i.interaction_type == InteractionType.DELAYED)
        confused = sum(1 for i in interactions if i.interaction_type == InteractionType.CONFUSED)
        memory_issues = sum(1 for i in interactions if i.memory_issue_detected)
        
        # Cognitive risk scores
        risk_scores = [i.cognitive_risk_score for i in interactions]
        avg_risk = statistics.mean(risk_scores)
        peak_risk = max(risk_scores)
        lowest_risk = min(risk_scores)
        
        # Response times
        response_times = [i.response_time_seconds for i in interactions if i.response_time_seconds]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Time of day analysis
        hour_performance = defaultdict(list)
        for interaction in interactions:
            hour = interaction.interaction_time.hour
            completed = interaction.interaction_type == InteractionType.CONFIRMED
            hour_performance[hour].append(completed)
        
        hour_completion_rates = {
            hour: sum(completions) / len(completions)
            for hour, completions in hour_performance.items()
        }
        
        sorted_hours = sorted(hour_completion_rates.items(), key=lambda x: x[1], reverse=True)
        best_hours = [h for h, _ in sorted_hours[:3]] if sorted_hours else []
        worst_hours = [h for h, _ in sorted_hours[-3:]] if sorted_hours else []
        
        # Category breakdown
        category_stats = defaultdict(lambda: {
            'total': 0, 'completed': 0, 'confused': 0, 'avg_risk': []
        })
        
        for interaction in interactions:
            cat = interaction.reminder_category or 'unknown'
            category_stats[cat]['total'] += 1
            if interaction.interaction_type == InteractionType.CONFIRMED:
                category_stats[cat]['completed'] += 1
            if interaction.interaction_type == InteractionType.CONFUSED:
                category_stats[cat]['confused'] += 1
            category_stats[cat]['avg_risk'].append(interaction.cognitive_risk_score)
        
        category_breakdown = {}
        for cat, data in category_stats.items():
            category_breakdown[cat] = {
                'total': data['total'],
                'completed': data['completed'],
                'confused': data['confused'],
                'completion_rate': data['completed'] / data['total'] if data['total'] > 0 else 0,
                'avg_risk': statistics.mean(data['avg_risk']) if data['avg_risk'] else 0
            }
        
        # Alert statistics
        total_alerts = len(alerts)
        critical_alerts = sum(1 for a in alerts if a.severity == 'critical')
        high_alerts = sum(1 for a in alerts if a.severity == 'high')
        unresolved_alerts = sum(1 for a in alerts if not a.is_resolved)
        
        return {
            'total_reminders': total,
            'completed_reminders': confirmed,
            'missed_reminders': ignored,
            'completion_rate': confirmed / total if total > 0 else 0,
            'avg_cognitive_risk': avg_risk,
            'peak_risk': peak_risk,
            'lowest_risk': lowest_risk,
            'confusion_count': confused,
            'memory_issue_count': memory_issues,
            'confirmed_count': confirmed,
            'ignored_count': ignored,
            'delayed_count': delayed,
            'best_hours': best_hours,
            'worst_hours': worst_hours,
            'avg_response_time': avg_response_time,
            'category_breakdown': category_breakdown,
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'high_priority_alerts': high_alerts,
            'unresolved_alerts': unresolved_alerts
        }
    
    def _generate_daily_summaries(
        self,
        interactions: List[ReminderInteraction],
        alerts: List[CaregiverAlert],
        start_date: datetime,
        end_date: datetime
    ) -> List[DailyRiskSummary]:
        """Generate daily summaries for the week."""
        daily_summaries = []
        
        current_date = start_date.date()
        end = end_date.date()
        
        while current_date <= end:
            day_interactions = [
                i for i in interactions
                if i.interaction_time.date() == current_date
            ]
            
            day_alerts = [
                a for a in alerts
                if a.created_at.date() == current_date
            ]
            
            if day_interactions:
                risk_scores = [i.cognitive_risk_score for i in day_interactions]
                avg_risk = statistics.mean(risk_scores)
                confusion_count = sum(1 for i in day_interactions if i.interaction_type == InteractionType.CONFUSED)
                completed = sum(1 for i in day_interactions if i.interaction_type == InteractionType.CONFIRMED)
                completion_rate = completed / len(day_interactions)
            else:
                avg_risk = 0
                confusion_count = 0
                completion_rate = 0
            
            daily_summaries.append(DailyRiskSummary(
                date=current_date.isoformat(),
                avg_cognitive_risk=avg_risk,
                confusion_count=confusion_count,
                total_interactions=len(day_interactions),
                completion_rate=completion_rate,
                alert_count=len(day_alerts)
            ))
            
            current_date += timedelta(days=1)
        
        return daily_summaries
    
    def _analyze_risk_trend(self, daily_summaries: List[DailyRiskSummary]) -> str:
        """Analyze trend in cognitive risk over the week."""
        if len(daily_summaries) < 3:
            return "insufficient_data"
        
        risk_scores = [d.avg_cognitive_risk for d in daily_summaries if d.total_interactions > 0]
        
        if len(risk_scores) < 3:
            return "stable"
        
        # Calculate trend using linear regression slope
        x = list(range(len(risk_scores)))
        y = risk_scores
        
        n = len(x)
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Determine trend based on slope
        if slope < -0.05:
            return "improving"
        elif slope > 0.05:
            return "declining"
        else:
            return "stable"
    
    def _get_previous_week_comparison(
        self,
        user_id: str,
        current_start: datetime
    ) -> Dict[str, Optional[float]]:
        """Get previous week statistics for comparison."""
        try:
            prev_end = current_start - timedelta(days=1)
            prev_start = prev_end - timedelta(days=7)
            
            prev_interactions = self._get_interactions(user_id, prev_start, prev_end)
            
            if not prev_interactions:
                return {'avg_risk': None, 'change_percentage': None}
            
            prev_risk_scores = [i.cognitive_risk_score for i in prev_interactions]
            prev_avg_risk = statistics.mean(prev_risk_scores)
            
            # This will be filled when current week avg is available
            return {'avg_risk': prev_avg_risk, 'change_percentage': None}
            
        except Exception as e:
            logger.error(f"Error getting previous week comparison: {e}")
            return {'avg_risk': None, 'change_percentage': None}
    
    def _generate_recommendations(
        self,
        stats: Dict[str, Any],
        risk_trend: str,
        alerts: List[CaregiverAlert]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Completion rate recommendations
        if stats['completion_rate'] < 0.6:
            recommendations.append(
                "[WARNING] Low completion rate detected. Consider adjusting reminder frequency and timing."
            )
        
        # Cognitive risk recommendations
        if stats['avg_cognitive_risk'] > 0.7:
            recommendations.append(
                "🚨 High average cognitive risk. Recommend immediate medical evaluation."
            )
        elif stats['avg_cognitive_risk'] > 0.5:
            recommendations.append(
                "[WARNING] Elevated cognitive risk. Schedule follow-up with healthcare provider."
            )
        
        # Trend recommendations
        if risk_trend == "declining":
            recommendations.append(
                "📉 Cognitive performance declining. Increase caregiver monitoring and consider medication adjustment."
            )
        elif risk_trend == "improving":
            recommendations.append(
                " Cognitive performance improving. Continue current care plan."
            )
        
        # Confusion recommendations
        if stats['confusion_count'] > stats['total_reminders'] * 0.3:
            recommendations.append(
                "💭 Frequent confusion detected. Simplify reminders and add more context/visual cues."
            )
        
        # Time-based recommendations
        if stats['worst_hours']:
            worst_hours_str = ", ".join(f"{h}:00" for h in stats['worst_hours'])
            recommendations.append(
                f"⏰ Poor performance during {worst_hours_str}. Reschedule critical reminders to optimal times."
            )
        
        if stats['best_hours']:
            best_hours_str = ", ".join(f"{h}:00" for h in stats['best_hours'])
            recommendations.append(
                f"[SUCCESS] Best performance at {best_hours_str}. Schedule important reminders during these hours."
            )
        
        # Alert recommendations
        if stats['unresolved_alerts'] > 3:
            recommendations.append(
                f"🔔 {stats['unresolved_alerts']} unresolved alerts. Immediate caregiver follow-up required."
            )
        
        # Category-specific recommendations
        for category, data in stats['category_breakdown'].items():
            if data['completion_rate'] < 0.5 and data['total'] >= 3:
                recommendations.append(
                    f"📋 Low completion for {category} reminders ({data['completion_rate']:.0%}). Review timing and content."
                )
        
        if not recommendations:
            recommendations.append(
                "[SUCCESS] Patient performance is stable. Continue current monitoring schedule."
            )
        
        return recommendations
    
    def _determine_risk_level(self, avg_cognitive_risk: float) -> str:
        """Determine overall risk level category."""
        if avg_cognitive_risk >= 0.75:
            return "critical"
        elif avg_cognitive_risk >= 0.6:
            return "high"
        elif avg_cognitive_risk >= 0.4:
            return "moderate"
        else:
            return "low"
    
    def _needs_intervention(
        self,
        stats: Dict[str, Any],
        risk_trend: str,
        alerts: List[CaregiverAlert]
    ) -> bool:
        """Determine if intervention is needed."""
        # High cognitive risk
        if stats['avg_cognitive_risk'] > 0.6:
            return True
        
        # Declining trend with moderate risk
        if risk_trend == "declining" and stats['avg_cognitive_risk'] > 0.4:
            return True
        
        # High confusion rate
        if stats['confusion_count'] > stats['total_reminders'] * 0.4:
            return True
        
        # Multiple unresolved alerts
        if stats['unresolved_alerts'] > 2:
            return True
        
        # Low completion rate
        if stats['completion_rate'] < 0.5:
            return True
        
        return False
    
    def _needs_escalation(
        self,
        stats: Dict[str, Any],
        alerts: List[CaregiverAlert]
    ) -> bool:
        """Determine if escalation to medical professional is needed."""
        # Critical cognitive risk
        if stats['avg_cognitive_risk'] >= 0.75:
            return True
        
        # Multiple critical alerts
        if stats['critical_alerts'] >= 3:
            return True
        
        # Very high confusion rate
        if stats['confusion_count'] > stats['total_reminders'] * 0.5:
            return True
        
        # Peak risk very high
        if stats['peak_risk'] >= 0.85:
            return True
        
        return False
    
    def _default_statistics(self) -> Dict[str, Any]:
        """Return default statistics when no data available."""
        return {
            'total_reminders': 0,
            'completed_reminders': 0,
            'missed_reminders': 0,
            'completion_rate': 0,
            'avg_cognitive_risk': 0,
            'peak_risk': 0,
            'lowest_risk': 0,
            'confusion_count': 0,
            'memory_issue_count': 0,
            'confirmed_count': 0,
            'ignored_count': 0,
            'delayed_count': 0,
            'best_hours': [],
            'worst_hours': [],
            'avg_response_time': 0,
            'category_breakdown': {},
            'total_alerts': 0,
            'critical_alerts': 0,
            'high_priority_alerts': 0,
            'unresolved_alerts': 0
        }
    
    def _empty_report(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> WeeklyCognitiveReport:
        """Return empty report when no data available."""
        return WeeklyCognitiveReport(
            user_id=user_id,
            report_period_start=start_date.isoformat(),
            report_period_end=end_date.isoformat(),
            generated_at=datetime.now().isoformat(),
            total_reminders=0,
            completed_reminders=0,
            missed_reminders=0,
            completion_rate=0,
            avg_cognitive_risk_score=0,
            peak_cognitive_risk_score=0,
            lowest_cognitive_risk_score=0,
            risk_trend="insufficient_data",
            confusion_count=0,
            memory_issue_count=0,
            confirmed_count=0,
            ignored_count=0,
            delayed_count=0,
            daily_summaries=[],
            total_alerts=0,
            critical_alerts=0,
            high_priority_alerts=0,
            unresolved_alerts=0,
            best_response_hours=[],
            worst_response_hours=[],
            avg_response_time_seconds=0,
            category_breakdown={},
            risk_level="unknown",
            recommendations=["No data available for this period."],
            intervention_needed=False,
            escalation_required=False
        )
    
    def export_report_to_pdf(self, report: WeeklyCognitiveReport, output_path: str):
        """
        Export report to PDF format.
        
        Args:
            report: WeeklyCognitiveReport to export
            output_path: Path to save PDF file
        """
        # TODO: Implement PDF generation using reportlab or similar
        logger.info(f"PDF export not yet implemented. Report saved to: {output_path}")
        pass
    
    def export_report_to_json(self, report: WeeklyCognitiveReport, output_path: str):
        """
        Export report to JSON format.
        
        Args:
            report: WeeklyCognitiveReport to export
            output_path: Path to save JSON file
        """
        import json
        
        with open(output_path, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        logger.info(f"Report exported to JSON: {output_path}")
