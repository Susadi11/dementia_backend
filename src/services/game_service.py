# src/services/game_service.py
"""
Async Game Service: Orchestrates the complete pipeline
Compatible with Motor (async MongoDB)
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
from src.database import Database  # Use teammate's async Database

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]
DEFAULT_MOTOR_BASELINE = 0.5  # seconds
LSTM_WINDOW_SIZE = 10  # last N sessions for LSTM

# ============================================================================
# Helper: Get or Create Motor Baseline (ASYNC)
# ============================================================================
async def get_motor_baseline(userId: str) -> float:
    """
    Retrieve user's motor baseline from calibrations collection.
    ASYNC version using Motor.
    """
    calibrations = Database.get_collection("calibrations")
    
    calibration = await calibrations.find_one(
        {"userId": userId},
        sort=[("calibrationDate", -1)]  # most recent
    )
    
    if calibration:
        return calibration.get("motorBaseline", DEFAULT_MOTOR_BASELINE)
    
    logger.warning(f"No calibration found for user {userId}, using default motor baseline")
    return DEFAULT_MOTOR_BASELINE

# ============================================================================
# Helper: Fetch Last N Sessions (ASYNC)
# ============================================================================
async def fetch_last_n_sessions(userId: str, n: int) -> List[Dict]:
    """
    Fetch last N game sessions for a user (for LSTM temporal analysis).
    ASYNC version.
    """
    game_sessions = Database.get_collection("game_sessions")
    
    cursor = game_sessions.find(
        {"userId": userId}
    ).sort("timestamp", -1).limit(n)
    
    sessions = await cursor.to_list(length=n)
    
    # Reverse to get chronological order (oldest → newest)
    sessions.reverse()
    
    return sessions

# ============================================================================
# Helper: Extract Features for LSTM
# ============================================================================
def extract_lstm_features(sessions: List[Dict]) -> np.ndarray:
    """
    Convert session history into LSTM input format.
    (Same as before - no DB calls)
    """
    if not sessions:
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
    
    X = np.array([feature_vectors])
    return X

# ============================================================================
# Helper: Run LSTM Inference
# ============================================================================
def predict_lstm_decline(sessions: List[Dict]) -> float:
    """
    Run LSTM model on session history to detect decline trend.
    (Same as before - no changes needed)
    """
    if len(sessions) < 3:
        return 0.0
    
    X = extract_lstm_features(sessions)
    
    lstm_model = get_lstm_model_safe()
    scaler = get_lstm_scaler()
    
    if scaler is not None:
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat)
        X = X_scaled.reshape(original_shape)
    
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
    (Same as before - no changes needed)
    """
    if not sessions:
        return np.array([[
            current_features["sac"],
            0.0,
            current_features["ies"],
            0.0,
            current_features["accuracy"],
            current_features["rtAdjMedian"],
            current_features["variability"],
            lstm_score,
            current_features["sac"],
            current_features["ies"]
        ]])
    
    sac_values = [s["features"]["sac"] for s in sessions]
    ies_values = [s["features"]["ies"] for s in sessions]
    acc_values = [s["features"]["accuracy"] for s in sessions]
    rt_values = [s["features"]["rtAdjMedian"] for s in sessions]
    var_values = [s["features"]["variability"] for s in sessions]
    
    mean_sac = np.mean(sac_values)
    mean_ies = np.mean(ies_values)
    mean_acc = np.mean(acc_values)
    mean_rt = np.mean(rt_values)
    mean_var = np.mean(var_values)
    
    def compute_slope(values):
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    slope_sac = compute_slope(sac_values)
    slope_ies = compute_slope(ies_values)
    
    features = [
        mean_sac, slope_sac, mean_ies, slope_ies,
        mean_acc, mean_rt, mean_var, lstm_score,
        current_features["sac"], current_features["ies"]
    ]
    
    return np.array([features])

# ============================================================================
# Helper: Run Risk Classification
# ============================================================================
def predict_risk(sessions: List[Dict], current_features: Dict, lstm_score: float) -> Dict:
    """
    Run risk classifier to get risk probabilities and level.
    (Same as before - no changes needed)
    """
    X = extract_risk_features(sessions, current_features, lstm_score)
    
    risk_model = get_risk_classifier_safe()
    scaler = get_risk_scaler()
    
    if scaler is not None:
        X = scaler.transform(X)
    
    try:
        probs = risk_model.predict_proba(X)[0]
        
        risk_probability = {
            "LOW": round(float(probs[0]), 4),
            "MEDIUM": round(float(probs[1]), 4),
            "HIGH": round(float(probs[2]), 4)
        }
        
        predicted_class = int(np.argmax(probs))
        risk_level = RISK_LABELS[predicted_class]
        risk_score_0_100 = round(float(probs[2]) * 100, 2)
        
        return {
            "riskProbability": risk_probability,
            "riskLevel": risk_level,
            "riskScore0_100": risk_score_0_100,
            "lstmDeclineScore": round(lstm_score, 4)
        }
        
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        return {
            "riskProbability": {"LOW": 0.33, "MEDIUM": 0.34, "HIGH": 0.33},
            "riskLevel": "MEDIUM",
            "riskScore0_100": 33.0,
            "lstmDeclineScore": lstm_score
        }

# ============================================================================
# Main Pipeline Function (ASYNC)
# ============================================================================
async def process_game_session(
    userId: str,
    sessionId: str,
    gameType: str,
    level: int,
    trials: Optional[List[Dict]] = None,
    summary: Optional[Dict] = None
) -> Dict:
    """
    Complete pipeline for processing a game session.
    ASYNC version compatible with Motor.
    """
    logger.info(f"Processing session {sessionId} for user {userId}")
    
    # Step 1: Get motor baseline (ASYNC)
    motor_baseline = await get_motor_baseline(userId)
    logger.info(f"Motor baseline: {motor_baseline}s")
    
    # Step 2: Compute features (sync - no DB calls)
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
    
    # Step 3: Fetch last N sessions for temporal analysis (ASYNC)
    past_sessions = await fetch_last_n_sessions(userId, LSTM_WINDOW_SIZE)
    logger.info(f"Fetched {len(past_sessions)} past sessions")
    
    # Step 4: Run LSTM inference (sync - model inference)
    lstm_score = predict_lstm_decline(past_sessions)
    logger.info(f"LSTM decline score: {lstm_score}")
    
    # Step 5: Run risk classification (sync - model inference)
    prediction = predict_risk(past_sessions, features, lstm_score)
    logger.info(f"Risk prediction: {prediction['riskLevel']} (score: {prediction['riskScore0_100']})")
    
    # Step 6: Store in MongoDB (ASYNC)
    game_sessions = Database.get_collection("game_sessions")
    
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
    
    await game_sessions.insert_one(session_doc)
    logger.info(f"Session stored in database")
    
    # Step 7: Check if alert needed (ASYNC)
    if prediction["riskLevel"] == "HIGH":
        alerts = Database.get_collection("alerts")
        alert = {
            "userId": userId,
            "alertType": "RISK_INCREASE",
            "message": f"High dementia risk detected (score: {prediction['riskScore0_100']})",
            "severity": "HIGH",
            "timestamp": datetime.utcnow(),
            "acknowledged": False
        }
        await alerts.insert_one(alert)
        logger.info("Alert created for HIGH risk")
    
    # Return response
    return {
        "sessionId": sessionId,
        "userId": userId,
        "features": features,
        "prediction": prediction,
        "timestamp": session_doc["timestamp"].isoformat()
    }