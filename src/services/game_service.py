# src/services/game_service.py
"""
Game Service: Orchestrates the complete pipeline
- Load calibration
- Compute features (SAC, IES, motor adjustment)
- Fetch session history
- Run LSTM inference
- Run risk classification
- Store results in MongoDB
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from src.features.game.cognitive_scoring import (
    compute_session_features,
    compute_features_from_summary
)
from src.models.game.model_registry import (
    get_lstm_model_safe,
    get_lstm_scaler,
    get_risk_classifier_safe,
    get_risk_scaler
)
from src.utils.database import get_db

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]
DEFAULT_MOTOR_BASELINE = 0.5  # seconds
LSTM_WINDOW_SIZE = 10  # last N sessions for LSTM

# ============================================================================
# Helper: Get or Create Motor Baseline
# ============================================================================
def get_motor_baseline(userId: str, db) -> float:
    """
    Retrieve user's motor baseline from calibrations collection.
    If not found, return default.
    """
    calibration = db.calibrations.find_one(
        {"userId": userId},
        sort=[("calibrationDate", -1)]  # most recent
    )
    
    if calibration:
        return calibration.get("motorBaseline", DEFAULT_MOTOR_BASELINE)
    
    logger.warning(f"No calibration found for user {userId}, using default motor baseline")
    return DEFAULT_MOTOR_BASELINE

# ============================================================================
# Helper: Fetch Last N Sessions
# ============================================================================
def fetch_last_n_sessions(userId: str, n: int, db) -> List[Dict]:
    """
    Fetch last N game sessions for a user (for LSTM temporal analysis).
    
    Returns:
        List of session documents, sorted by timestamp (oldest to newest)
    """
    sessions = list(
        db.game_sessions.find(
            {"userId": userId}
        )
        .sort("timestamp", -1)  # newest first
        .limit(n)
    )
    
    # Reverse to get chronological order (oldest → newest)
    sessions.reverse()
    
    return sessions

# ============================================================================
# Helper: Extract Features for LSTM
# ============================================================================
def extract_lstm_features(sessions: List[Dict]) -> np.ndarray:
    """
    Convert session history into LSTM input format.
    
    Expected LSTM input shape: (batch_size, timesteps, features)
    For single prediction: (1, N_sessions, N_features)
    
    Features per session:
        - SAC
        - IES
        - accuracy
        - rtAdjMedian
        - variability
    
    Args:
        sessions: List of session dictionaries
        
    Returns:
        X: Array of shape (1, len(sessions), 5) for LSTM
    """
    if not sessions:
        # Return zeros if no history (will be handled by risk classifier)
        return np.zeros((1, 1, 5))
    
    feature_vectors = []
    
    for session in sessions:
        features = session.get("features", {})
        
        vec = [
            features.get("sac", 0.0),
            features.get("ies", 0.0),
            features.get("accuracy", 0.0),
            features.get("rtAdjMedian", 0.0),
            features.get("variability", 0.0)
        ]
        feature_vectors.append(vec)
    
    # Shape: (1, n_sessions, 5)
    X = np.array([feature_vectors])
    
    return X

# ============================================================================
# Helper: Run LSTM Inference
# ============================================================================
def predict_lstm_decline(sessions: List[Dict]) -> float:
    """
    Run LSTM model on session history to detect decline trend.
    
    Returns:
        decline_score: Float between 0-1 (higher = more decline detected)
    """
    if len(sessions) < 3:
        # Not enough history for LSTM
        return 0.0
    
    # Extract features
    X = extract_lstm_features(sessions)
    
    # Load model
    lstm_model = get_lstm_model_safe()
    scaler = get_lstm_scaler()
    
    # Scale if scaler available
    if scaler is not None:
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat)
        X = X_scaled.reshape(original_shape)
    
    # Predict
    try:
        prediction = lstm_model.predict(X, verbose=0)
        decline_score = float(prediction[0][0])
        return decline_score
    except Exception as e:
        logger.error(f"LSTM prediction failed: {e}")
        return 0.0

# ============================================================================
# Helper: Extract Aggregate Features for Risk Classifier
# ============================================================================
def extract_risk_features(sessions: List[Dict], current_features: Dict, lstm_score: float) -> np.ndarray:
    """
    Build feature vector for risk classifier.
    
    Features (example):
        1. mean_SAC_lastN
        2. slope_SAC_lastN (linear trend)
        3. mean_IES_lastN
        4. slope_IES_lastN
        5. mean_accuracy_lastN
        6. mean_rtAdj_lastN
        7. variability_rtAdj_lastN
        8. lstm_decline_score
        9. current_SAC
        10. current_IES
    
    Args:
        sessions: Past sessions (last N)
        current_features: Features from current session
        lstm_score: LSTM decline score
        
    Returns:
        X: Array of shape (1, n_features)
    """
    if not sessions:
        # No history, use only current features + lstm_score
        return np.array([[
            current_features["sac"],
            0.0,  # no trend
            current_features["ies"],
            0.0,
            current_features["accuracy"],
            current_features["rtAdjMedian"],
            current_features["variability"],
            lstm_score,
            current_features["sac"],
            current_features["ies"]
        ]])
    
    # Extract metrics from history
    sac_values = [s["features"]["sac"] for s in sessions]
    ies_values = [s["features"]["ies"] for s in sessions]
    acc_values = [s["features"]["accuracy"] for s in sessions]
    rt_values = [s["features"]["rtAdjMedian"] for s in sessions]
    var_values = [s["features"]["variability"] for s in sessions]
    
    # Compute means
    mean_sac = np.mean(sac_values)
    mean_ies = np.mean(ies_values)
    mean_acc = np.mean(acc_values)
    mean_rt = np.mean(rt_values)
    mean_var = np.mean(var_values)
    
    # Compute trends (linear slope)
    def compute_slope(values):
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    slope_sac = compute_slope(sac_values)
    slope_ies = compute_slope(ies_values)
    
    # Build feature vector
    features = [
        mean_sac,
        slope_sac,
        mean_ies,
        slope_ies,
        mean_acc,
        mean_rt,
        mean_var,
        lstm_score,
        current_features["sac"],
        current_features["ies"]
    ]
    
    return np.array([features])

# ============================================================================
# Helper: Run Risk Classification
# ============================================================================
def predict_risk(sessions: List[Dict], current_features: Dict, lstm_score: float) -> Dict:
    """
    Run risk classifier to get risk probabilities and level.
    
    Returns:
        {
            "riskProbability": {"LOW": 0.2, "MEDIUM": 0.3, "HIGH": 0.5},
            "riskLevel": "HIGH",
            "riskScore0_100": 50.0,
            "lstmDeclineScore": 0.25
        }
    """
    # Extract features
    X = extract_risk_features(sessions, current_features, lstm_score)
    
    # Load model
    risk_model = get_risk_classifier_safe()
    scaler = get_risk_scaler()
    
    # Scale if available
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict probabilities
    try:
        probs = risk_model.predict_proba(X)[0]  # [prob_low, prob_med, prob_high]
        
        # Build response
        risk_probability = {
            "LOW": round(float(probs[0]), 4),
            "MEDIUM": round(float(probs[1]), 4),
            "HIGH": round(float(probs[2]), 4)
        }
        
        # Determine label
        predicted_class = int(np.argmax(probs))
        risk_level = RISK_LABELS[predicted_class]
        
        # Compute 0-100 score
        risk_score_0_100 = round(float(probs[2]) * 100, 2)  # HIGH probability as score
        
        return {
            "riskProbability": risk_probability,
            "riskLevel": risk_level,
            "riskScore0_100": risk_score_0_100,
            "lstmDeclineScore": round(lstm_score, 4)
        }
        
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        # Return neutral default
        return {
            "riskProbability": {"LOW": 0.33, "MEDIUM": 0.34, "HIGH": 0.33},
            "riskLevel": "MEDIUM",
            "riskScore0_100": 33.0,
            "lstmDeclineScore": lstm_score
        }

# ============================================================================
# Main Pipeline Function
# ============================================================================
def process_game_session(
    userId: str,
    sessionId: str,
    gameType: str,
    level: int,
    trials: Optional[List[Dict]] = None,
    summary: Optional[Dict] = None
) -> Dict:
    """
    Complete pipeline for processing a game session.
    
    Args:
        userId: User identifier
        sessionId: Unique session ID
        gameType: Type of game (e.g., "card_matching")
        level: Difficulty level
        trials: List of trial data (preferred) OR
        summary: Summary metrics (fallback)
        
    Returns:
        Complete result dictionary with features + prediction
    """
    db = get_db()
    
    logger.info(f"Processing session {sessionId} for user {userId}")
    
    # Step 1: Get motor baseline
    motor_baseline = get_motor_baseline(userId, db)
    logger.info(f"Motor baseline: {motor_baseline}s")
    
    # Step 2: Compute features
    if trials is not None:
        features = compute_session_features(trials, motor_baseline)
        raw_summary = {
            "totalAttempts": len(trials),
            "correct": sum(t.get("correct", 0) for t in trials),
            "errors": len(trials) - sum(t.get("correct", 0) for t in trials),
            "hintsUsed": sum(t.get("hint_used", 0) for t in trials),
            "meanRtRaw": float(np.mean([t["rt_raw"] for t in trials])),
            "medianRtRaw": float(np.median([t["rt_raw"] for t in trials]))
        }
    elif summary is not None:
        features = compute_features_from_summary(
            summary["totalAttempts"],
            summary["correct"],
            summary["meanRtRaw"],
            motor_baseline
        )
        raw_summary = summary
    else:
        raise ValueError("Either 'trials' or 'summary' must be provided")
    
    logger.info(f"Features computed: SAC={features['sac']}, IES={features['ies']}")
    
    # Step 3: Fetch last N sessions for temporal analysis
    past_sessions = fetch_last_n_sessions(userId, LSTM_WINDOW_SIZE, db)
    logger.info(f"Fetched {len(past_sessions)} past sessions")
    
    # Step 4: Run LSTM inference
    lstm_score = predict_lstm_decline(past_sessions)
    logger.info(f"LSTM decline score: {lstm_score}")
    
    # Step 5: Run risk classification
    prediction = predict_risk(past_sessions, features, lstm_score)
    logger.info(f"Risk prediction: {prediction['riskLevel']} (score: {prediction['riskScore0_100']})")
    
    # Step 6: Store in MongoDB
    session_doc = {
        "userId": userId,
        "sessionId": sessionId,
        "timestamp": datetime.utcnow(),
        "gameType": gameType,
        "level": level,
        "motorBaseline": motor_baseline,
        "rawSummary": raw_summary,
        "features": features,
        "prediction": prediction,
        "createdAt": datetime.utcnow()
    }
    
    db.game_sessions.insert_one(session_doc)
    logger.info(f"Session stored in database")
    
    # Step 7: Check if alert needed (optional)
    if prediction["riskLevel"] == "HIGH":
        alert = {
            "userId": userId,
            "alertType": "RISK_INCREASE",
            "message": f"High dementia risk detected (score: {prediction['riskScore0_100']})",
            "severity": "HIGH",
            "timestamp": datetime.utcnow(),
            "acknowledged": False
        }
        db.alerts.insert_one(alert)
        logger.info("Alert created for HIGH risk")
    
    # Return response
    return {
        "sessionId": sessionId,
        "userId": userId,
        "features": features,
        "prediction": prediction,
        "timestamp": session_doc["timestamp"].isoformat()
    }