from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class WeeklyRiskCalculator:

    def __init__(self):
        # Max possible session score is 36
        self.max_session_score = 36

    async def calculate_weekly_risk(
        self,
        db,
        user_id: str,
        week_start: datetime
    ) -> Dict[str, Any]:
        # Define current and previous week boundaries
        week_end = week_start + timedelta(days=7) - timedelta(seconds=1)

        # Fetch sessions for current week
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

        # Fetch previous week sessions for trend
        prev_week_start = week_start - timedelta(days=7)
        prev_week_end = week_start - timedelta(seconds=1)
        previous_week_sessions = await self._get_week_sessions(
            db, user_id, prev_week_start, prev_week_end
        )

        # Calculate current week average and base score
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

        # Apply trend to get final risk score
        final_weekly_risk = self._calculate_final_risk(
            current_week_base_score, weekly_error_increase
        )

        # Classify risk level from final score
        risk_level = self._get_risk_level(final_weekly_risk)

        # Breakdown scores by time of day
        time_window_breakdown = self._calculate_time_window_breakdown(
            current_week_sessions
        )

        # Average Random Forest probability if available
        rf_weekly_avg = self._calculate_rf_average(current_week_sessions)

        result = {
            "user_id": user_id,
            "week_start": week_start,
            "week_end": week_end,
            "sessions_count": len(current_week_sessions),
            "weekly_avg_score": round(current_week_avg, 2),
            "weekly_base_score": round(current_week_base_score, 2),
            "previous_week_avg": round(previous_week_avg, 2) if previous_week_avg else None,
            "weekly_error_increase": round(weekly_error_increase, 2) if weekly_error_increase else None,
            "final_weekly_risk": round(final_weekly_risk, 2),
            "risk_level": risk_level,
            "time_window_breakdown": time_window_breakdown,
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
        # Query MongoDB for sessions in date range
        try:
            collection = db["chat_detection_sessions"]

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
        # WeeklyAvg = sum(raw scores) / num sessions
        if not sessions:
            return 0.0

        total_score = sum(
            session.get("session_raw_score", 0) for session in sessions
        )

        return total_score / len(sessions)

    def _normalize_to_100(self, weekly_avg: float) -> float:
        # WeeklyBaseScore = (WeeklyAvg / 36) × 100
        weekly_base_score = (weekly_avg / self.max_session_score) * 100
        return min(weekly_base_score, 100)

    def _calculate_trend(
        self,
        current_week_avg: float,
        previous_week_avg: float
    ) -> float:
        # WeeklyErrorIncrease = (current - previous) / previous × 100
        if previous_week_avg == 0:
            if current_week_avg > 0:
                return 100.0
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
        # FinalWeeklyRisk = BaseScore × (1 + ErrorIncrease/100), cap 100
        if weekly_error_increase is None:
            return weekly_base_score

        final_weekly_risk = weekly_base_score * (1 + (weekly_error_increase / 100))
        final_weekly_risk = max(0, min(final_weekly_risk, 100))

        return final_weekly_risk

    def _get_risk_level(self, final_risk: float) -> str:
        # Map 0-100 score to risk level string
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
        # Group session scores by time of day window
        time_windows = {
            "morning": [],
            "afternoon": [],
            "evening": [],
            "night": []
        }

        for session in sessions:
            window = session.get("time_window")
            score = session.get("session_raw_score", 0)
            if window in time_windows:
                time_windows[window].append(score)

        # Average each time window score
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
        # Average Random Forest probability across sessions
        rf_probabilities = [
            session.get("rf_probability")
            for session in sessions
            if session.get("rf_probability") is not None
        ]

        if not rf_probabilities:
            return None

        return sum(rf_probabilities) / len(rf_probabilities)
