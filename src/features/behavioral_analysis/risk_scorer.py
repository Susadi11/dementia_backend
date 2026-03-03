"""
Dementia Risk Scorer (Step 4)

Takes deviation results from ChronosAnalyzer and converts them into
a DementiaRiskReport with:

  - Risk level  : LOW / MEDIUM / HIGH
  - Actions     : increase reminders, alert caregiver
  - Empathy tone: feeds into susadi/hale-empathy-3b chatbot

Risk thresholds (tunable via environment variables):
  - LOW    : overall deviation < 20%
  - MEDIUM : 20% ≤ deviation < 50%  → double reminder frequency
  - HIGH   : deviation ≥ 50%         → alert caregiver immediately
"""

import logging
import os
from typing import Dict, Optional
from datetime import datetime

from src.features.behavioral_analysis.behavioral_models import (
    BehaviorDeviationResult,
    DementiaRiskLevel,
    DementiaRiskReport,
    BehavioralTimeSeries,
)
from src.features.behavioral_analysis.chronos_analyzer import ChronosAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (can be overridden via .env)
# ---------------------------------------------------------------------------

RISK_THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "20.0"))  # %
RISK_THRESHOLD_HIGH   = float(os.getenv("RISK_THRESHOLD_HIGH",   "50.0"))  # %

# Feature weights — some behaviors are stronger signals than others
FEATURE_WEIGHTS: Dict[str, float] = {
    "avg_completion_rate":          1.5,   # Strong signal
    "medication_misses":            2.0,   # Strongest: missing meds is critical
    "avg_medication_delay_minutes": 1.2,
    "avg_response_delay_minutes":   1.0,
    "app_interactions":             0.8,   # Weaker signal
}


class DementiaRiskScorer:
    """
    Orchestrates the full analysis pipeline:
      1. Run ChronosAnalyzer on all features
      2. Compute weighted deviation score
      3. Classify into LOW / MEDIUM / HIGH risk
      4. Generate recommended actions
      5. Set empathy tone for chatbot

    Usage:
        scorer = DementiaRiskScorer()
        report = await scorer.compute_risk(time_series, previous_report)
    """

    def __init__(self):
        self.analyzer = ChronosAnalyzer(forecast_horizon=7)

    async def compute_risk(
        self,
        time_series: BehavioralTimeSeries,
        previous_report: Optional[DementiaRiskReport] = None,
    ) -> DementiaRiskReport:
        """
        Full pipeline: behavioral data → DementiaRiskReport.

        Args:
            time_series      : User's behavioral time series from MongoDB
            previous_report  : Last report (for trend calculation)

        Returns:
            DementiaRiskReport with risk level, actions, and chatbot guidance
        """
        user_id = time_series.user_id
        logger.info(f"Computing dementia risk for user={user_id}")

        # Step 1: Run Chronos on all behavioral features
        deviation_results: Dict[str, BehaviorDeviationResult] = (
            self.analyzer.analyze_all_features(time_series)
        )

        # Step 2: Compute weighted overall deviation score
        weighted_score, feature_scores = self._compute_weighted_score(deviation_results)

        # Step 3: Classify risk level
        risk_level = self._classify_risk(weighted_score)

        # Step 4: Determine trend (vs previous report)
        trend = self._compute_trend(risk_level, previous_report)

        # Step 5: Generate actions and chatbot tone
        recommended_action, increase_reminders, alert_caregiver = (
            self._determine_actions(risk_level)
        )
        empathy_tone = self._determine_empathy_tone(risk_level, trend)

        report = DementiaRiskReport(
            user_id=user_id,
            generated_at=datetime.utcnow(),
            risk_level=risk_level,
            deviation_percentage=round(weighted_score, 2),
            feature_scores=feature_scores,
            recommended_action=recommended_action,
            increase_reminder_frequency=increase_reminders,
            alert_caregiver=alert_caregiver,
            empathy_tone=empathy_tone,
            previous_risk_level=previous_report.risk_level if previous_report else None,
            trend=trend,
        )

        logger.info(
            f"Risk report for user={user_id} | "
            f"level={risk_level} deviation={weighted_score:.1f}% trend={trend}"
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_weighted_score(
        self,
        results: Dict[str, BehaviorDeviationResult]
    ) -> tuple:
        """
        Apply feature weights and compute a single overall deviation %.
        Returns (weighted_score, feature_scores_dict).
        """
        total_weight = 0.0
        weighted_sum = 0.0
        feature_scores: Dict[str, float] = {}

        for feature, result in results.items():
            weight = FEATURE_WEIGHTS.get(feature, 1.0)
            score  = result.deviation_percentage
            feature_scores[feature] = round(score, 2)
            weighted_sum += score * weight
            total_weight += weight

        overall = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        return overall, feature_scores

    def _classify_risk(self, deviation_pct: float) -> DementiaRiskLevel:
        """
        Map deviation percentage to risk level.

        LOW    < 20%  → routine is stable
        MEDIUM 20-50% → some disruption, increase reminders
        HIGH   ≥ 50%  → significant disruption, alert caregiver
        """
        if deviation_pct < RISK_THRESHOLD_MEDIUM:
            return DementiaRiskLevel.LOW
        elif deviation_pct < RISK_THRESHOLD_HIGH:
            return DementiaRiskLevel.MEDIUM
        else:
            return DementiaRiskLevel.HIGH

    def _compute_trend(
        self,
        current: DementiaRiskLevel,
        previous: Optional[DementiaRiskReport]
    ) -> str:
        """Compare current risk level to previous to detect trend."""
        if previous is None:
            return "stable"

        level_order = {
            DementiaRiskLevel.LOW:    0,
            DementiaRiskLevel.MEDIUM: 1,
            DementiaRiskLevel.HIGH:   2,
        }
        curr_val = level_order[current]
        prev_val = level_order[previous.risk_level]

        if curr_val < prev_val:
            return "improving"
        elif curr_val > prev_val:
            return "declining"
        return "stable"

    def _determine_actions(
        self,
        risk_level: DementiaRiskLevel
    ) -> tuple:
        """
        Returns (recommended_action, increase_reminder_frequency, alert_caregiver)
        based on risk level.
        """
        if risk_level == DementiaRiskLevel.LOW:
            return (
                "Continue current reminder schedule. Routine appears stable.",
                False,
                False,
            )
        elif risk_level == DementiaRiskLevel.MEDIUM:
            return (
                "Behavioral deviation detected. Increasing reminder frequency "
                "and sending gentle check-in messages.",
                True,
                False,
            )
        else:  # HIGH
            return (
                "Significant behavioral deviation detected. "
                "Caregiver has been notified. Reminders set to maximum frequency.",
                True,
                True,
            )

    def _determine_empathy_tone(
        self,
        risk_level: DementiaRiskLevel,
        trend: str
    ) -> str:
        """
        Determine the tone guidance for susadi/hale-empathy-3b.
        This is passed as a system prompt hint to the chatbot.

        Returns:
            "supportive"   – normal friendly conversation
            "gentle_alert" – caring but slightly more attentive
            "urgent"       – warm but proactive, involve caregiver
        """
        if risk_level == DementiaRiskLevel.LOW:
            return "supportive"
        elif risk_level == DementiaRiskLevel.MEDIUM:
            return "gentle_alert"
        else:
            return "urgent"
