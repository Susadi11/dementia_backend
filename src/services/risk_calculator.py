"""
Weekly Risk Calculator

Calculates weekly dementia risk score using the refined equation:
1. Calculate weekly average session score
2. Normalize to 0-100
3. Apply trend vs previous week
4. Cap at 100
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WeeklyRiskCalculator:
    """
    Calculates weekly risk using session data.

    Equation:
    WeeklyAvg = sum(session_raw_scores) / num_sessions
    WeeklyBaseScore = (WeeklyAvg / 36) × 100
    WeeklyErrorIncrease = (CurrentWeek - PreviousWeek) / PreviousWeek × 100
    FinalWeeklyRisk = WeeklyBaseScore × (1 + WeeklyErrorIncrease/100)
    Cap at 100
    """

    def __init__(self):
        """Initialize risk calculator"""
        self.max_session_score = 36

    async def calculate_weekly_risk(
        self,
        db,
        user_id: str,
        week_start: datetime
    ) -> Dict[str, Any]:
        """
        Calculate weekly risk for a user.

        Args:
            db: MongoDB database instance
            user_id: User ID
            week_start: Start date of the week (datetime)

        Returns:
            Dictionary with weekly risk metrics
        """
        # Define week boundaries
        week_end = week_start + timedelta(days=7) - timedelta(seconds=1)

        # Get current week sessions
        current_week_sessions = await self._get_week_sessions(
            db, user_id, week_start, week_end
        )

        if not current_week_sessions:
            logger.warning(f"No sessions found for user {user_id} in week {week_start}")
            return {
                "error": "No sessions found for this week",
                "user_id": user_id,
                "week_start": week_start,
                "week_end": week_end,
                "sessions_count": 0
            }

        # Get previous week sessions (for trend calculation)
        prev_week_start = week_start - timedelta(days=7)
        prev_week_end = week_start - timedelta(seconds=1)
        previous_week_sessions = await self._get_week_sessions(
            db, user_id, prev_week_start, prev_week_end
        )

        # Calculate current week metrics
        current_week_avg = self._calculate_weekly_average(current_week_sessions)
        current_week_base_score = self._normalize_to_100(current_week_avg)

        # Calculate trend vs previous week
        weekly_error_increase = None
        previous_week_avg = None

        if previous_week_sessions:
            previous_week_avg = self._calculate_weekly_average(previous_week_sessions)
            weekly_error_increase = self._calculate_trend(
                current_week_avg, previous_week_avg
            )

        # Calculate final weekly risk
        final_weekly_risk = self._calculate_final_risk(
            current_week_base_score, weekly_error_increase
        )

        # Determine risk level
        risk_level = self._get_risk_level(final_weekly_risk)

        # Calculate time window breakdown
        time_window_breakdown = self._calculate_time_window_breakdown(
            current_week_sessions
        )

        # Calculate RF average if available
        rf_weekly_avg = self._calculate_rf_average(current_week_sessions)

        result = {
            "user_id": user_id,
            "week_start": week_start,
            "week_end": week_end,
            "sessions_count": len(current_week_sessions),

            # Weekly metrics
            "weekly_avg_score": round(current_week_avg, 2),
            "weekly_base_score": round(current_week_base_score, 2),

            # Trend
            "previous_week_avg": round(previous_week_avg, 2) if previous_week_avg else None,
            "weekly_error_increase": round(weekly_error_increase, 2) if weekly_error_increase else None,

            # Final risk (0-100)
            "final_weekly_risk": round(final_weekly_risk, 2),
            "risk_level": risk_level,

            # Time window breakdown
            "time_window_breakdown": time_window_breakdown,

            # RF validation
            "rf_weekly_avg": round(rf_weekly_avg, 3) if rf_weekly_avg else None,

            "calculated_at": datetime.now()
        }

        logger.info(
            f"Weekly risk calculated for {user_id}: {final_weekly_risk:.2f} ({risk_level})"
        )

        return result

    async def _get_week_sessions(
        self,
        db,
        user_id: str,
        week_start: datetime,
        week_end: datetime
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all sessions for a user within a week.

        Args:
            db: MongoDB database instance
            user_id: User ID
            week_start: Week start date
            week_end: Week end date

        Returns:
            List of session documents
        """
        try:
            collection = db["detection_sessions"]

            query = {
                "user_id": user_id,
                "timestamp": {
                    "$gte": week_start,
                    "$lte": week_end
                }
            }

            cursor = collection.find(query).sort("timestamp", 1)
            sessions = await cursor.to_list(length=None)

            logger.info(
                f"Retrieved {len(sessions)} sessions for {user_id} "
                f"between {week_start} and {week_end}"
            )

            return sessions

        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            raise

    def _calculate_weekly_average(self, sessions: List[Dict[str, Any]]) -> float:
        """
        Calculate weekly average session score.

        WeeklyAvg = sum(session_raw_scores) / num_sessions

        Args:
            sessions: List of session documents

        Returns:
            Weekly average score (0-36)
        """
        if not sessions:
            return 0.0

        total_score = sum(
            session.get("session_raw_score", 0) for session in sessions
        )

        weekly_avg = total_score / len(sessions)

        return weekly_avg

    def _normalize_to_100(self, weekly_avg: float) -> float:
        """
        Normalize weekly average to 0-100 scale.

        WeeklyBaseScore = (WeeklyAvg / 36) × 100

        Args:
            weekly_avg: Average session score (0-36)

        Returns:
            Normalized score (0-100)
        """
        weekly_base_score = (weekly_avg / self.max_session_score) * 100

        return min(weekly_base_score, 100)

    def _calculate_trend(
        self,
        current_week_avg: float,
        previous_week_avg: float
    ) -> float:
        """
        Calculate trend (weekly error increase).

        WeeklyErrorIncrease = (CurrentWeek - PreviousWeek) / PreviousWeek × 100

        Args:
            current_week_avg: Current week average (0-36)
            previous_week_avg: Previous week average (0-36)

        Returns:
            Percentage change
        """
        if previous_week_avg == 0:
            # Avoid division by zero
            if current_week_avg > 0:
                return 100.0  # 100% increase from 0
            return 0.0

        weekly_error_increase = (
            (current_week_avg - previous_week_avg) / previous_week_avg
        ) * 100

        return weekly_error_increase

    def _calculate_final_risk(
        self,
        weekly_base_score: float,
        weekly_error_increase: Optional[float]
    ) -> float:
        """
        Calculate final weekly risk with trend adjustment.

        FinalWeeklyRisk = WeeklyBaseScore × (1 + WeeklyErrorIncrease/100)
        Cap at 100

        Args:
            weekly_base_score: Base score (0-100)
            weekly_error_increase: Trend percentage (can be negative)

        Returns:
            Final risk score (0-100)
        """
        if weekly_error_increase is None:
            # No previous week data - return base score
            return weekly_base_score

        # Apply trend factor
        final_weekly_risk = weekly_base_score * (1 + (weekly_error_increase / 100))

        # Cap at 100 (and minimum 0)
        final_weekly_risk = max(0, min(final_weekly_risk, 100))

        return final_weekly_risk

    def _get_risk_level(self, final_risk: float) -> str:
        """
        Determine risk level from final risk score.

        0-20: Normal
        21-40: Mild
        41-60: Moderate
        61-80: High
        81-100: Critical

        Args:
            final_risk: Final risk score (0-100)

        Returns:
            Risk level string
        """
        if final_risk <= 20:
            return "Normal"
        elif final_risk <= 40:
            return "Mild"
        elif final_risk <= 60:
            return "Moderate"
        elif final_risk <= 80:
            return "High"
        else:
            return "Critical"

    def _calculate_time_window_breakdown(
        self,
        sessions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate average scores by time window.
        Shows sundowning effect.

        Args:
            sessions: List of session documents

        Returns:
            Dictionary with averages for each time window
        """
        time_windows = {
            "morning": [],
            "afternoon": [],
            "evening": [],
            "night": []
        }

        # Group sessions by time window
        for session in sessions:
            window = session.get("time_window")
            score = session.get("session_raw_score", 0)

            if window in time_windows:
                time_windows[window].append(score)

        # Calculate averages
        breakdown = {}
        for window, scores in time_windows.items():
            if scores:
                breakdown[f"{window}_avg"] = round(sum(scores) / len(scores), 2)
                breakdown[f"{window}_count"] = len(scores)
            else:
                breakdown[f"{window}_avg"] = None
                breakdown[f"{window}_count"] = 0

        return breakdown

    def _calculate_rf_average(self, sessions: List[Dict[str, Any]]) -> Optional[float]:
        """
        Calculate average Random Forest probability (if available).

        Args:
            sessions: List of session documents

        Returns:
            Average RF probability or None
        """
        rf_probabilities = [
            session.get("rf_probability")
            for session in sessions
            if session.get("rf_probability") is not None
        ]

        if not rf_probabilities:
            return None

        return sum(rf_probabilities) / len(rf_probabilities)
