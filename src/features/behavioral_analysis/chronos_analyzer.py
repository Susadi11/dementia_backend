"""
Chronos-T5-Small Behavioral Analyzer (Step 3)

Uses amazon/chronos-t5-small to:
  1. Take a user's past behavioral time-series as input
  2. Predict what their expected behavior should look like
  3. Compare actual vs predicted → deviation = cognitive risk signal

Model: amazon/chronos-t5-small
  - Zero-shot time-series forecasting
  - No retraining required
  - Works on any numerical time series

How it works here:
  - Input  : 30 days of daily behavioral metrics (e.g., avg_completion_rate)
  - Output : Predicted values for the next window
  - Deviation : |actual - predicted| → risk signal
"""

import logging
import statistics
from typing import List, Optional, Tuple
import numpy as np

from src.features.behavioral_analysis.behavioral_models import (
    BehavioralTimeSeries,
    BehaviorDeviationResult,
    DailyBehaviorSummary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy global model instance
# ---------------------------------------------------------------------------
_chronos_pipeline = None


def _get_chronos_pipeline():
    """
    Load amazon/chronos-t5-small model once and reuse across requests.
    Uses CPU by default (matching existing NLP_DEVICE=cpu setting).
    """
    global _chronos_pipeline
    if _chronos_pipeline is None:
        try:
            import torch
            from chronos import ChronosPipeline  # pip install chronos-forecasting

            logger.info("Loading amazon/chronos-t5-small model...")
            _chronos_pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("✅ Chronos-T5-Small loaded successfully")

        except ImportError:
            logger.warning(
                "chronos-forecasting not installed. "
                "Falling back to statistical baseline predictor."
            )
            _chronos_pipeline = "FALLBACK"

        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            _chronos_pipeline = "FALLBACK"

    return _chronos_pipeline


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

ANALYZABLE_FEATURES = {
    "avg_completion_rate":          lambda d: d.avg_completion_rate,
    "avg_medication_delay_minutes": lambda d: d.avg_medication_delay_minutes,
    "medication_misses":            lambda d: float(d.medication_misses),
    "avg_response_delay_minutes":   lambda d: d.avg_response_delay_minutes,
    "app_interactions":             lambda d: float(d.app_interactions),
}


def _extract_series(
    days: List[DailyBehaviorSummary],
    feature: str
) -> List[float]:
    """Extract a list of floats from daily summaries for a given feature."""
    extractor = ANALYZABLE_FEATURES.get(feature)
    if extractor is None:
        raise ValueError(f"Unknown feature: {feature}")
    return [extractor(d) for d in days]


# ---------------------------------------------------------------------------
# Statistical fallback predictor
# ---------------------------------------------------------------------------

def _statistical_predict(history: List[float], horizon: int = 7) -> List[float]:
    """
    Simple moving-average baseline when Chronos is unavailable.
    Uses last-14-day mean as the predicted value.
    """
    window = history[-14:] if len(history) >= 14 else history
    mean = statistics.mean(window) if window else 0.5
    return [mean] * horizon


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class ChronosAnalyzer:
    """
    Wraps the Chronos-T5-Small model for behavioral deviation analysis.

    Usage:
        analyzer = ChronosAnalyzer()
        result = analyzer.analyze(time_series, feature="avg_completion_rate")
    """

    def __init__(self, forecast_horizon: int = 7):
        """
        Args:
            forecast_horizon: How many days ahead to predict (default 7).
        """
        self.forecast_horizon = forecast_horizon

    def analyze(
        self,
        time_series: BehavioralTimeSeries,
        feature: str = "avg_completion_rate"
    ) -> BehaviorDeviationResult:
        """
        Main method: runs Chronos on the given time series and returns
        a BehaviorDeviationResult with predicted vs actual comparison.

        Workflow:
          1. Extract numerical series from BehavioralTimeSeries
          2. Split into history (training context) and last-N-days actual
          3. Feed history to Chronos → get predicted values
          4. Compare predicted vs actual → compute deviation %
        """
        days = time_series.days
        user_id = time_series.user_id

        if len(days) < 7:
            logger.warning(
                f"User {user_id} has only {len(days)} days of data. "
                "Need at least 7 for analysis. Returning zero deviation."
            )
            return BehaviorDeviationResult(
                user_id=user_id,
                feature_analyzed=feature,
                actual_values=[],
                predicted_values=[],
                mean_absolute_error=0.0,
                deviation_percentage=0.0,
            )

        full_series = _extract_series(days, feature)

        # Split: use all but last 7 days as context; last 7 as "actual"
        context_len = max(len(full_series) - self.forecast_horizon, 7)
        context_values = full_series[:context_len]
        actual_values  = full_series[context_len:]

        # Predict
        predicted_values = self._predict(context_values, len(actual_values))

        # Compute deviation
        mae, deviation_pct = self._compute_deviation(actual_values, predicted_values, feature)

        logger.info(
            f"Chronos analysis | user={user_id} feature={feature} "
            f"deviation={deviation_pct:.1f}% MAE={mae:.4f}"
        )

        return BehaviorDeviationResult(
            user_id=user_id,
            feature_analyzed=feature,
            actual_values=actual_values,
            predicted_values=predicted_values,
            mean_absolute_error=round(mae, 4),
            deviation_percentage=round(deviation_pct, 2),
        )

    def _predict(
        self,
        context: List[float],
        horizon: int
    ) -> List[float]:
        """Run Chronos or fall back to statistical predictor."""
        pipeline = _get_chronos_pipeline()

        if pipeline == "FALLBACK":
            return _statistical_predict(context, horizon)

        try:
            import torch

            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            # Chronos returns (num_samples, horizon) shaped tensor
            forecast = pipeline.predict(context_tensor, prediction_length=horizon)
            # Use the median (50th percentile) across samples
            median_forecast = forecast[0].median(dim=0).values.tolist()
            return median_forecast

        except Exception as e:
            logger.error(f"Chronos prediction failed: {e}. Using fallback.")
            return _statistical_predict(context, horizon)

    def _compute_deviation(
        self,
        actual: List[float],
        predicted: List[float],
        feature: str
    ) -> Tuple[float, float]:
        """
        Compute MAE and deviation percentage.

        For features where higher = worse (e.g., medication_misses, delays),
        a positive deviation still means things got worse.
        """
        if not actual or not predicted:
            return 0.0, 0.0

        n = min(len(actual), len(predicted))
        actual_arr    = np.array(actual[:n])
        predicted_arr = np.array(predicted[:n])

        mae = float(np.mean(np.abs(actual_arr - predicted_arr)))

        # Normalize by predicted range to get a % deviation
        pred_range = max(float(np.max(np.abs(predicted_arr))), 1e-6)
        deviation_pct = (mae / pred_range) * 100.0

        # Cap at 100%
        deviation_pct = min(deviation_pct, 100.0)

        return mae, deviation_pct

    def analyze_all_features(
        self,
        time_series: BehavioralTimeSeries
    ) -> dict:
        """
        Run analysis on all tracked features and return a dict of
        feature_name → BehaviorDeviationResult.
        Used by the risk scorer to get a comprehensive picture.
        """
        results = {}
        for feature in ANALYZABLE_FEATURES:
            try:
                results[feature] = self.analyze(time_series, feature)
            except Exception as e:
                logger.error(f"Failed to analyze feature {feature}: {e}")
        return results
