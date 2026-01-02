# src/features/game/cognitive_scoring.py
"""
Cognitive scoring functions: SAC, IES, Motor Adjustment
These are pure functions with no database or model dependencies
"""
import numpy as np
from typing import List, Dict

# ============================================================================
# Motor Baseline Calibration
# ============================================================================
def compute_motor_baseline(tap_times: List[float]) -> float:
    """
    Compute motor baseline from a simple reaction time task.
    
    Args:
        tap_times: List of reaction times (seconds) from calibration taps
        
    Returns:
        motor_baseline: Median tap time (robust to outliers)
        
    Example:
        tap_times = [0.3, 0.32, 0.29, 0.31, 0.28, 0.45]  # one outlier
        baseline = compute_motor_baseline(tap_times)  # ~0.30s
    """
    if not tap_times or len(tap_times) == 0:
        return 0.5  # default fallback
    
    return float(np.median(tap_times))

# ============================================================================
# Motor-Adjusted Reaction Time
# ============================================================================
def adjust_reaction_time(rt_raw: float, motor_baseline: float, epsilon: float = 0.05) -> float:
    """
    Remove motor component from raw reaction time.
    
    Args:
        rt_raw: Raw measured reaction time (seconds)
        motor_baseline: User's calibrated motor delay (seconds)
        epsilon: Minimum adjusted RT to prevent negative/zero values
        
    Returns:
        rt_adj: Motor-adjusted reaction time (cognitive component)
        
    Formula:
        RT_adj = max(RT_raw - motor_baseline, epsilon)
    """
    rt_adj = rt_raw - motor_baseline
    return max(rt_adj, epsilon)

def adjust_reaction_times_batch(rt_raw_list: List[float], motor_baseline: float) -> List[float]:
    """Batch version for multiple trials"""
    return [adjust_reaction_time(rt, motor_baseline) for rt in rt_raw_list]

# ============================================================================
# Session Metrics Computation
# ============================================================================
def compute_session_accuracy(correct: int, total_attempts: int) -> float:
    """
    Compute accuracy for a session.
    
    Returns:
        accuracy: Between 0.0 and 1.0
    """
    if total_attempts == 0:
        return 0.0
    return correct / total_attempts

def compute_session_rt_adj_median(rt_adj_list: List[float]) -> float:
    """
    Compute median adjusted RT for a session (robust to outliers).
    """
    if not rt_adj_list or len(rt_adj_list) == 0:
        return 1.0  # fallback
    return float(np.median(rt_adj_list))

def compute_rt_variability(rt_adj_list: List[float]) -> float:
    """
    Compute standard deviation of adjusted RTs (variability indicator).
    High variability may indicate inconsistent attention.
    """
    if not rt_adj_list or len(rt_adj_list) < 2:
        return 0.0
    return float(np.std(rt_adj_list))

# ============================================================================
# SAC (Speed-Accuracy Composite)
# ============================================================================
def compute_sac(accuracy: float, rt_adj_median: float) -> float:
    """
    Speed-Accuracy Composite: Rewards both accuracy and efficiency.
    
    Formula:
        SAC = Accuracy / RT_adj_median
        
    Interpretation:
        - Higher SAC = better cognitive efficiency
        - SAC decreases if accuracy drops or RT increases
        
    Args:
        accuracy: Session accuracy (0.0 to 1.0)
        rt_adj_median: Median motor-adjusted RT (seconds)
        
    Returns:
        sac: Efficiency score (higher is better)
        
    Example:
        accuracy = 0.80, rt_adj = 1.4s → SAC = 0.571
    """
    if rt_adj_median <= 0:
        return 0.0
    
    return accuracy / rt_adj_median

# ============================================================================
# IES (Inverse Efficiency Score)
# ============================================================================
def compute_ies(rt_adj_median: float, accuracy: float, min_accuracy: float = 0.1) -> float:
    """
    Inverse Efficiency Score: Time cost per correct response.
    
    Formula:
        IES = RT_adj_median / max(Accuracy, min_accuracy)
        
    Interpretation:
        - Lower IES = better efficiency
        - IES increases if RT increases or accuracy decreases
        
    Args:
        rt_adj_median: Median motor-adjusted RT (seconds)
        accuracy: Session accuracy (0.0 to 1.0)
        min_accuracy: Clamp to prevent division explosion
        
    Returns:
        ies: Inverse efficiency (lower is better)
        
    Example:
        rt_adj = 1.4s, accuracy = 0.80 → IES = 1.75
    """
    accuracy_clamped = max(accuracy, min_accuracy)
    return rt_adj_median / accuracy_clamped

# ============================================================================
# Complete Session Feature Computation
# ============================================================================
def compute_session_features(
    trials: List[Dict],  # List of trial data: [{rt_raw, correct, error}, ...]
    motor_baseline: float
) -> Dict:
    """
    Compute all features for a game session.
    
    Args:
        trials: List of trial dictionaries with keys:
            - rt_raw: raw reaction time (seconds)
            - correct: 1 if correct, 0 otherwise
            - error: 1 if error, 0 otherwise
        motor_baseline: User's motor baseline (seconds)
        
    Returns:
        Dictionary with:
            - accuracy
            - errorRate
            - rtAdjMedian
            - sac
            - ies
            - variability
    """
    if not trials or len(trials) == 0:
        return {
            "accuracy": 0.0,
            "errorRate": 0.0,
            "rtAdjMedian": 1.0,
            "sac": 0.0,
            "ies": 10.0,
            "variability": 0.0
        }
    
    # Extract data
    rt_raw_list = [t.get("rt_raw", 1.0) for t in trials]
    correct_list = [t.get("correct", 0) for t in trials]
    
    # Adjust RTs
    rt_adj_list = adjust_reaction_times_batch(rt_raw_list, motor_baseline)
    
    # Compute metrics
    total_attempts = len(trials)
    correct_count = sum(correct_list)
    error_count = total_attempts - correct_count
    
    accuracy = compute_session_accuracy(correct_count, total_attempts)
    error_rate = error_count / total_attempts if total_attempts > 0 else 0.0
    rt_adj_median = compute_session_rt_adj_median(rt_adj_list)
    variability = compute_rt_variability(rt_adj_list)
    
    sac = compute_sac(accuracy, rt_adj_median)
    ies = compute_ies(rt_adj_median, accuracy)
    
    return {
        "accuracy": round(accuracy, 4),
        "errorRate": round(error_rate, 4),
        "rtAdjMedian": round(rt_adj_median, 4),
        "sac": round(sac, 4),
        "ies": round(ies, 4),
        "variability": round(variability, 4)
    }

# ============================================================================
# Alternative: If frontend sends summary instead of trials
# ============================================================================
def compute_features_from_summary(
    total_attempts: int,
    correct: int,
    mean_rt_raw: float,
    motor_baseline: float
) -> Dict:
    """
    Simplified version when only summary metrics are available.
    
    Note: This loses some accuracy (can't compute median, variability properly)
    but is useful if frontend only sends aggregated data.
    """
    accuracy = compute_session_accuracy(correct, total_attempts)
    error_rate = (total_attempts - correct) / total_attempts if total_attempts > 0 else 0.0
    
    # Approximate adjusted RT using mean
    rt_adj_approx = max(mean_rt_raw - motor_baseline, 0.05)
    
    sac = compute_sac(accuracy, rt_adj_approx)
    ies = compute_ies(rt_adj_approx, accuracy)
    
    return {
        "accuracy": round(accuracy, 4),
        "errorRate": round(error_rate, 4),
        "rtAdjMedian": round(rt_adj_approx, 4),
        "sac": round(sac, 4),
        "ies": round(ies, 4),
        "variability": 0.0  # cannot compute without trial-level data
    }