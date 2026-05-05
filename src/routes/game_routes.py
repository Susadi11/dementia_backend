# src/routes/game_routes.py
"""
Game API Routes
Endpoints for game session processing, calibration, and history
Compatible with teammate's async Database class
"""
from fastapi import APIRouter, HTTPException, status, Depends, Header
from datetime import datetime
from typing import List, Optional
import logging

from src.parsers.game_schemas import (
    GameSessionRequest,
    GameSessionResponse,
    CalibrationRequest,
    CalibrationResponse,
    SessionHistoryResponse,
    SessionHistoryItem,
    UserStatsResponse
)
from src.services.game_service import process_game_session
from src.features.game.cognitive_scoring import compute_motor_baseline
from src.database import Database  # Use teammate's Database class
from src.routes.user_routes import get_current_user
from src.routes.caregiver_routes import get_current_caregiver
from src.utils.auth import verify_token

router = APIRouter(prefix="/game", tags=["Game"])
logger = logging.getLogger(__name__)

# Helper: Look up assigned caregiver ID for a user
async def lookup_caregiver_id_for_user(userId: str) -> Optional[str]:
    """
    Get the caregiver_id assigned to a user by reading it directly from
    the user's own document (users collection has caregiver_id field).
    Returns the caregiver_id string or None if not found.
    """
    try:
        users = Database.get_collection("users")
        user_doc = await users.find_one({"user_id": userId}, {"caregiver_id": 1})
        if user_doc:
            return user_doc.get("caregiver_id") or None
    except Exception as e:
        logger.warning(f"Could not look up caregiver for user {userId}: {e}")
    return None


async def verify_caregiver_linked(caregiver_id: str, user_id: str):
    """
    Raise 403 if the caregiver does not have user_id in their patient_ids.
    """
    caregivers = Database.get_collection("caregivers")
    caregiver = await caregivers.find_one(
        {"caregiver_id": caregiver_id, "patient_ids": user_id},
        {"caregiver_id": 1}
    )
    if not caregiver:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: user {user_id} is not linked to your caregiver account"
        )


# POST /game/session - Process Game Session
@router.post("/session", response_model=GameSessionResponse, status_code=status.HTTP_201_CREATED)
async def submit_game_session(
    request: GameSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Always use the authenticated user's ID from the JWT token
        user_id = current_user["user_id"]
        logger.info(f"Game session from authenticated user: {user_id}")

        # Resolve caregiverId: use provided value or look up from DB
        caregiver_id = request.caregiverId
        if not caregiver_id:
            caregiver_id = await lookup_caregiver_id_for_user(user_id)
            if caregiver_id:
                logger.info(f"Auto-resolved caregiverId={caregiver_id} for userId={user_id}")
            else:
                logger.warning(f"No caregiver found for userId={user_id}")

        # Convert trials to dict if provided
        trials = None
        if request.trials is not None and len(request.trials) > 0:
            trials = [t.model_dump(by_alias=False) for t in request.trials]

            # AUTO-FIX: Convert milliseconds to seconds if RT values are too large
            # Expected RT: 0.5-3.0 seconds. If RT > 10, assume it's in milliseconds
            if trials and any(t.get('rt_raw', 0) > 10 for t in trials):
                logger.warning(f"⚠️ RT values appear to be in milliseconds, converting to seconds")
                for t in trials:
                    if t.get('rt_raw', 0) > 10:
                        t['rt_raw'] = t['rt_raw'] / 1000.0
                logger.info(f"✓ Converted RT from ms to seconds: {[t['rt_raw'] for t in trials[:3]]}")

        # Convert summary to dict if provided
        summary = None
        if request.summary is not None:
            summary = request.summary.model_dump()

            # AUTO-FIX: Convert milliseconds to seconds for summary too
            if summary.get('meanRtRaw', 0) > 10:
                logger.warning(f"⚠️ Mean RT appears to be in milliseconds ({summary['meanRtRaw']}), converting to seconds")
                summary['meanRtRaw'] = summary['meanRtRaw'] / 1000.0
                if summary.get('medianRtRaw'):
                    summary['medianRtRaw'] = summary['medianRtRaw'] / 1000.0
                logger.info(f"✓ Converted mean RT: {summary['meanRtRaw']:.3f}s")

        # Process session (ASYNC) — userId from JWT, not request body
        result = await process_game_session(
            userId=user_id,
            caregiverId=caregiver_id,
            sessionId=request.sessionId,
            gameType=request.gameType,
            level=request.level,
            trials=trials,
            summary=summary
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# GET /game/motor-baseline  (auth-based — reads userId from JWT token)
# Auth is optional: if no valid token is provided, returns a no-calibration
# response gracefully (handles 307 redirect from empty-userId frontend calls).
# ============================================================================
@router.get("/motor-baseline")
async def get_motor_baseline_from_token(
    authorization: Optional[str] = Header(None)
):
    # If no token provided (e.g. trailing-slash redirect stripped the header), return empty
    if not authorization:
        return {"userId": None, "motor_baseline": None, "message": "No auth token provided"}

    try:
        token = authorization.replace("Bearer ", "").strip()
        payload = verify_token(token)
    except Exception:
        return {"userId": None, "motor_baseline": None, "message": "Invalid token"}

    if payload is None:
        return {"userId": None, "motor_baseline": None, "message": "Invalid or expired token"}

    userId = payload.get("user_id")
    if not userId:
        return {"userId": None, "motor_baseline": None, "message": "User not found in token"}

    try:
        calibrations = Database.get_collection("calibrations")
        calibration = await calibrations.find_one(
            {"userId": userId},
            sort=[("calibrationDate", -1)]
        )

        if not calibration:
            return {"userId": userId, "motor_baseline": None, "message": "No calibration found"}

        return {
            "userId": userId,
            "motor_baseline": calibration.get("motorBaseline"),
            "n_taps": len(calibration.get("tapTimes", [])),
            "created_at": calibration.get("calibrationDate")
        }

    except Exception as e:
        logger.error(f"Error fetching baseline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch baseline")


# GET /game/motor-baseline/{userId}  (kept for backward compatibility)
@router.get("/motor-baseline/{userId}")
async def get_motor_baseline(userId: str):
    """
    Get user's latest motor baseline.
    """
    try:
        calibrations = Database.get_collection("calibrations")
        calibration = await calibrations.find_one(
            {"userId": userId},
            sort=[("calibrationDate", -1)]
        )
        
        if not calibration:
            return {"userId": userId, "motor_baseline": None, "message": "No calibration found"}
            
        return {
            "userId": userId,
            "motor_baseline": calibration.get("motorBaseline"),
            "n_taps": len(calibration.get("tapTimes", [])),
            "created_at": calibration.get("calibrationDate")
        }
        
    except Exception as e:
        logger.error(f"Error fetching baseline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch baseline")


# ===========================================================================
# POST /game/calibration - Motor Baseline Calibration
@router.post("/calibration", response_model=CalibrationResponse)
async def calibrate_motor_baseline(request: CalibrationRequest):
   
    try:
        calibrations = Database.get_collection("calibrations")
        
        # Compute motor baseline
        motor_baseline = compute_motor_baseline(request.tapTimes)
        
        # Store in database (ASYNC)
        calibration_doc = {
            "userId": request.userId,
            "motorBaseline": motor_baseline,
            "calibrationDate": datetime.utcnow(),
            "tapTimes": request.tapTimes
        }
        
        await calibrations.insert_one(calibration_doc)
        
        logger.info(f"Motor baseline calibrated for user {request.userId}: {motor_baseline}s")
        
        return CalibrationResponse(
            userId=request.userId,
            motorBaseline=motor_baseline,
            calibrationDate=calibration_doc["calibrationDate"].isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in calibration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Calibration failed")

# ============================================================================
# GET /game/history/{userId} - Get Session History
# ============================================================================
@router.get("/history/{userId}", response_model=SessionHistoryResponse)
async def get_session_history(userId: str, limit: int = 20):
    try:
        game_sessions = Database.get_collection("game_sessions")
        
        cursor = game_sessions.find({"userId": userId}).sort("timestamp", -1).limit(limit)
        sessions = await cursor.to_list(length=limit)
        
        history_items = []
        for session in sessions:
            features = session.get("features", {})
            prediction = session.get("prediction", {})
            raw_summary = session.get("rawSummary", {})

            # Use total accuracy (correct incl. hints / total) for display.
            # features["accuracy"] is correct-without-hints only (per thesis formula).
            # Fall back to features["accuracy"] when rawSummary is absent (old sessions).
            total_attempts = raw_summary.get("totalAttempts", 0)
            if total_attempts and total_attempts > 0:
                total_correct = raw_summary.get("correct", 0)
                total_accuracy = round(total_correct / total_attempts, 4)
                hints_used = raw_summary.get("hintsUsed", 0)
                hint_dep_rate = round(hints_used / total_attempts, 4)
            else:
                # Old session without rawSummary — use stored feature values
                total_accuracy = round(features.get("accuracy", 0.0), 4)
                hint_dep_rate = round(features.get("hintDependencyRate", 0.0), 4)

            # Floor riskScore for old sessions where rule override set level to HIGH/MEDIUM
            # but left riskScore0_100 at the ML model's near-zero value.
            risk_level_stored = prediction.get("riskLevel", "UNKNOWN")
            risk_score = prediction.get("riskScore0_100", 0.0)
            if risk_level_stored == "HIGH":
                risk_score = max(risk_score, 70.0)
            elif risk_level_stored == "MEDIUM":
                risk_score = max(risk_score, 35.0)

            item = SessionHistoryItem(
                sessionId=session["sessionId"],
                timestamp=session["timestamp"].isoformat(),
                gameType=session.get("gameType", "card_matching"),
                level=session.get("level", 1),
                accuracy=total_accuracy,
                hintDependencyRate=hint_dep_rate,
                sac=features.get("sac", 0.0),
                ies=features.get("ies", 0.0),
                riskLevel=risk_level_stored,
                riskScore=risk_score
            )
            history_items.append(item)
        
        return SessionHistoryResponse(
            userId=userId,
            totalSessions=len(history_items),
            sessions=history_items
        )
        
    except Exception as e:
        logger.error(f"Error fetching history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch history")

# GET /game/stats/{userId} - Get User Statistics
@router.get("/stats/{userId}", response_model=UserStatsResponse)
async def get_user_stats(userId: str):
    """
    Get aggregate statistics for a user (for dashboard overview).
    
    **Output:**
    - Total sessions
    - Average SAC/IES
    - Current risk level
    - Most recent risk score
    """
    try:
        game_sessions = Database.get_collection("game_sessions")
        
        cursor = game_sessions.find({"userId": userId}).sort("timestamp", -1)
        sessions = await cursor.to_list(length=None)
        
        if not sessions:
            # New user with no sessions — return clean zero stats so the frontend
            # clears any previously cached data instead of keeping old user's stats.
            # Return "" (not None) for lastSessionDate so Kotlin's non-nullable
            # String field deserializes without a crash.
            return UserStatsResponse(
                userId=userId,
                totalSessions=0,
                avgSAC=0.0,
                avgIES=0.0,
                currentRiskLevel="UNKNOWN",
                recentRiskScore=0.0,
                lastSessionDate=""
            )
        
        # Compute averages – use .get() so old sessions without a 'features' key don't crash
        total = len(sessions)
        avg_sac = sum(s.get("features", {}).get("sac", 0.0) for s in sessions) / total
        avg_ies = sum(s.get("features", {}).get("ies", 0.0) for s in sessions) / total
        
        # Get most recent
        latest = sessions[0]
        current_risk = latest["prediction"].get("riskLevel", "UNKNOWN")
        recent_score = latest["prediction"].get("riskScore0_100", 0.0)
        # Floor score for old sessions where rule override didn't update riskScore0_100
        if current_risk == "HIGH":
            recent_score = max(recent_score, 70.0)
        elif current_risk == "MEDIUM":
            recent_score = max(recent_score, 35.0)
        last_session = latest["timestamp"].isoformat()
        
        return UserStatsResponse(
            userId=userId,
            totalSessions=total,
            avgSAC=round(avg_sac, 4),
            avgIES=round(avg_ies, 4),
            currentRiskLevel=current_risk,
            recentRiskScore=recent_score,
            lastSessionDate=last_session
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute statistics")

# DELETE /game/session/{sessionId} - Delete Session (Optional)
@router.delete("/session/{sessionId}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(sessionId: str):
    """
    Delete a game session (for testing/cleanup).
    """
    try:
        game_sessions = Database.get_collection("game_sessions")
        result = await game_sessions.delete_one({"sessionId": sessionId})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Deleted session {sessionId}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session")

# GET /game/test-model - Test model prediction
@router.get("/test-model")
async def test_model_prediction():
    """
    Test endpoint to verify the risk classifier model is working.
    Simulates a low accuracy game (6%) and returns what the model predicts.
    """
    from src.models.game.model_registry import get_risk_classifier, get_risk_scaler
    from src.services.game_service import predict_risk
    import numpy as np
    
    try:
        # Get model info
        risk_model = get_risk_classifier()
        scaler = get_risk_scaler()
        
        model_info = {
            "model_type": risk_model.__class__.__name__ if risk_model else "None",
            "scaler_type": scaler.__class__.__name__ if scaler else "None"
        }
        
        # Simulate features from a 6% accuracy game
        simulated_features = {
            "accuracy": 0.06,  # 6% accuracy (3/50)
            "sac": 0.0266,
            "ies": 15.025,
            "rtAdjMedian": 2000,
            "variability": 0.8
        }
        
        # Make prediction
        prediction = predict_risk([], simulated_features, 0.0)
        
        return {
            "test_scenario": "6% accuracy (3/50 correct)",
            "model_info": model_info,
            "simulated_features": simulated_features,
            "prediction": prediction,
            "expected": "Should predict HIGH risk for 6% accuracy",
            "status": "✅ Model working!" if prediction["riskLevel"] == "HIGH" else "⚠️ Unexpected prediction"
        }
        
    except Exception as e:
        logger.error(f"Model test failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "❌ Model test failed"
        }