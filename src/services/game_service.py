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
# Note: RISK_LABELS order MUST match the label encoder's classes order!
# The label encoder has classes in alphabetical order: ['HIGH', 'LOW', 'MEDIUM']
RISK_LABELS = ["HIGH", "LOW", "MEDIUM"]  # Matches label encoder order
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
    Extracts 14 features to match the trained model:
    1-2: mean_sac, slope_sac
    3-4: mean_ies, slope_ies  
    5: mean_accuracy
    6: mean_rt
    7: mean_variability
    8: lstm_decline_score
    9-10: current_sac, current_ies
    11-12: slope_accuracy, slope_rt (NEW)
    13-14: std_sac, std_ies (NEW)
    """
    
    logger.info(f"🔍 EXTRACTING FEATURES - Accuracy: {current_features.get('accuracy', 0):.1%}, SAC: {current_features.get('sac', 0):.4f}, IES: {current_features.get('ies', 0):.4f}")
    
    if not sessions:
        features = [
            current_features["sac"],
            0.0,
            current_features["ies"],
            0.0,
            current_features["accuracy"],
            current_features["rtAdjMedian"],
            current_features["variability"],
            lstm_score,
            current_features["sac"],
            current_features["ies"],
            0.0,  # slope_accuracy
            0.0,  # slope_rt
            0.0,  # std_sac
            0.0   # std_ies
        ]
        logger.info(f"📊 FEATURES (first session): {features}")
        return np.array([features])
    
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
    
    std_sac = np.std(sac_values) if len(sac_values) > 1 else 0.0
    std_ies = np.std(ies_values) if len(ies_values) > 1 else 0.0
    
    def compute_slope(values):
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    slope_sac = compute_slope(sac_values)
    slope_ies = compute_slope(ies_values)
    slope_acc = compute_slope(acc_values)
    slope_rt = compute_slope(rt_values)
    
    features = [
        mean_sac, slope_sac, mean_ies, slope_ies,
        mean_acc, mean_rt, mean_var, lstm_score,
        current_features["sac"], current_features["ies"],
        slope_acc, slope_rt, std_sac, std_ies
    ]
    
    return np.array([features])

# ============================================================================
# Helper: Run Risk Classification
# ============================================================================
def predict_risk(sessions: List[Dict], current_features: Dict, lstm_score: float) -> Dict:
    """
    Run risk classifier to get risk probabilities and level.
    Uses the trained logistic regression model - NO FALLBACK/DUMMY LOGIC.
    """
    X = extract_risk_features(sessions, current_features, lstm_score)
    
    risk_model = get_risk_classifier_safe()
    scaler = get_risk_scaler()
    
    # Log which model is being used
    model_name = risk_model.__class__.__name__
    if model_name == 'DummyRiskClassifier':
        logger.error("❌❌❌ CRITICAL: Using DUMMY risk classifier - predictions will be RANDOM!")
        logger.error("❌ Model failed to load! Check model files immediately!")
        raise RuntimeError("Risk classifier model not loaded - cannot make predictions")
    else:
        logger.info(f"✓ Using trained model: {model_name}")
    
    # Log features being used
    logger.info(f"Input features: accuracy={current_features.get('accuracy', 0):.2%}, "
                f"sac={current_features.get('sac', 0):.4f}, "
                f"ies={current_features.get('ies', 0):.4f}, "
                f"rt={current_features.get('rtAdjMedian', 0):.0f}ms")
    
    if scaler is not None:
        X_raw = X.copy()
        X = scaler.transform(X)
        logger.info(f"✓ Features scaled (raw[0]={X_raw[0,0]:.4f} -> scaled[0]={X[0,0]:.4f})")
    else:
        logger.warning("⚠️ No scaler available - using raw features")
    
    try:
        probs = risk_model.predict_proba(X)[0]
        
        # ⚠️ TEMPORARY FIX: Model was trained with inverted labels!
        # The model predicts LOW for poor performance (should be HIGH)
        # Swap HIGH and LOW probabilities until model is retrained
        probs_fixed = np.array([probs[1], probs[0], probs[2]])  # Swap HIGH<->LOW
        logger.warning("⚠️ TEMP FIX: Inverting HIGH/LOW predictions (model trained with wrong labels)")
        
        # Map probabilities to correct labels (after inversion fix)
        # probs_fixed[0] = HIGH (was LOW), probs_fixed[1] = LOW (was HIGH), probs_fixed[2] = MEDIUM
        risk_probability = {
            "LOW": round(float(probs_fixed[1]), 4),
            "MEDIUM": round(float(probs_fixed[2]), 4),
            "HIGH": round(float(probs_fixed[0]), 4)
        }
        
        predicted_class = int(np.argmax(probs_fixed))
        risk_level = RISK_LABELS[predicted_class]  # Will be HIGH, LOW, or MEDIUM
        risk_score_0_100 = round(float(probs_fixed[0]) * 100, 2)  # HIGH risk percentage
        
        logger.info(f"✅ PREDICTION (after fix): {risk_level} | Probs: HIGH={probs_fixed[0]:.3f}, LOW={probs_fixed[1]:.3f}, MED={probs_fixed[2]:.3f} | Score: {risk_score_0_100}/100")
        
        return {
            "riskProbability": risk_probability,
            "riskLevel": risk_level,
            "riskScore0_100": risk_score_0_100,
            "lstmDeclineScore": round(lstm_score, 4)
        }
        
    except Exception as e:
        logger.error(f"❌ Risk prediction FAILED with exception: {e}", exc_info=True)
        # DO NOT return fallback - raise the error so we know something is wrong
        raise RuntimeError(f"Risk prediction failed: {e}") from e

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


# ============================================================================
# Risk Prediction Service (Standalone)
# ============================================================================
async def analyze_risk_window(userId: str, window_size: int = 10) -> Dict:
    """
    Compute risk independent of a new game session.
    Fetches last N sessions, computes features, runs models.
    """
    # 1. Fetch history
    sessions = await fetch_last_n_sessions(userId, window_size)
    
    if not sessions:
        # No history, return default low risk
        return {
            "prediction": {
                "prob_low": 1.0, "prob_medium": 0.0, "prob_high": 0.0,
                "label": "LOW", "risk_score_0_100": 0.0
            },
            "features_used": {
                "mean_sac": 0.0, "slope_sac": 0.0, "mean_ies": 0.0, "slope_ies": 0.0,
                "mean_accuracy": 0.0, "mean_rt": 0.0, "mean_variability": 0.0,
                "lstm_score": 0.0
            },
            "window_size": 0
        }

    # 2. Prepare features
    # Use the most recent session as "current" for feature extraction
    latest_session = sessions[-1] # sessions are reversed in fetch_last_n_sessions (oldest -> newest)
    
    current_features = latest_session.get("features", {})
    
    # 3. Models
    lstm_score = predict_lstm_decline(sessions)
    
    # Compute feature values for return (debug/display)
    sac_values = [s["features"].get("sac", 0) for s in sessions]
    ies_values = [s["features"].get("ies", 0) for s in sessions]
    rt_values = [s["features"].get("rtAdjMedian", 0) for s in sessions]
    
    def get_slope(values):
        if len(values) < 2: return 0.0
        return float(np.polyfit(np.arange(len(values)), values, 1)[0])
        
    features_used = {
        "mean_sac": float(np.mean(sac_values)),
        "slope_sac": get_slope(sac_values),
        "mean_ies": float(np.mean(ies_values)),
        "slope_ies": get_slope(ies_values),
        "mean_accuracy": float(np.mean([s["features"].get("accuracy", 0) for s in sessions])),
        "mean_rt": float(np.mean(rt_values)),
        "mean_variability": float(np.mean([s["features"].get("variability", 0) for s in sessions])),
        "lstm_score": lstm_score
    }
    
    # Run prediction
    prediction_result = predict_risk(sessions, current_features, lstm_score)
    
    # Map to simpler structure
    probs = prediction_result["riskProbability"]
    prediction_detail = {
        "prob_low": probs["LOW"],
        "prob_medium": probs["MEDIUM"],
        "prob_high": probs["HIGH"],
        "label": prediction_result["riskLevel"],
        "risk_score_0_100": prediction_result["riskScore0_100"]
    }
    
    # 4. Store standalone prediction
    risk_predictions = Database.get_collection("risk_predictions")
    prediction_doc = {
        "userId": userId,
        "window_size": window_size,
        "features_used": features_used,
        "prediction": prediction_detail,
        "created_at": datetime.utcnow()
    }
    await risk_predictions.insert_one(prediction_doc)
    
    return {
        "user_id": userId,
        "window_size": len(sessions),
        "features_used": features_used,
        "prediction": prediction_detail,
        "created_at": prediction_doc["created_at"].isoformat()
    }

async def get_risk_history(userId: str) -> List[Dict]:
    """Get all past risk predictions for a user"""
    risk_predictions = Database.get_collection("risk_predictions")
    cursor = risk_predictions.find({"userId": userId}).sort("created_at", -1)
    
    history = []
    async for doc in cursor:
        history.append({
            "user_id": doc["userId"],
            "window_size": doc["window_size"],
            "features_used": doc["features_used"],
            "prediction": doc["prediction"],
            "scale_note": "Risk bands align to GDS 1 / 2-3 / 4-5 (not diagnosis)",
            "created_at": doc["created_at"].isoformat()
        })
        
    return history
