# src/routes/game_routes.py
"""
Game API Routes
Endpoints for game session processing, calibration, and history
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import List
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
from src.utils.database import get_db

router = APIRouter(prefix="/game", tags=["Game"])
logger = logging.getLogger(__name__)

# ============================================================================
# POST /game/session - Process Game Session
# ============================================================================
@router.post("/session", response_model=GameSessionResponse, status_code=status.HTTP_201_CREATED)
async def submit_game_session(request: GameSessionRequest):
    """
    Process a completed game session and return risk assessment.
    
    **Workflow:**
    1. Load user's motor baseline
    2. Compute cognitive features (SAC, IES, motor-adjusted RT)
    3. Fetch last N sessions for temporal analysis
    4. Run LSTM model for decline detection
    5. Run risk classifier for risk level
    6. Store results in database
    7. Return features + risk prediction
    
    **Input:**
    - `trials`: List of trial-level data (preferred) OR
    - `summary`: Aggregated session metrics (fallback)
    
    **Output:**
    - Session features (SAC, IES, accuracy, etc.)
    - Risk prediction (LOW/MEDIUM/HIGH + probabilities)
    """
    try:
        # Convert trials to dict if provided
        trials = None
        if request.trials:
            trials = [t.dict() for t in request.trials]
        
        # Convert summary to dict if provided
        summary = None
        if request.summary:
            summary = request.summary.dict()
        
        # Process session
        result = process_game_session(
            userId=request.userId,
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
# POST /game/calibration - Motor Baseline Calibration
# ============================================================================
@router.post("/calibration", response_model=CalibrationResponse)
async def calibrate_motor_baseline(request: CalibrationRequest):
    """
    Calibrate user's motor baseline using simple reaction time task.
    
    **Instructions for Frontend:**
    1. Show a simple stimulus (e.g., circle appears)
    2. User taps as quickly as possible
    3. Repeat 5-10 times
    4. Send all tap times to this endpoint
    
    **Example:**
    ```json
    {
      "userId": "user123",
      "tapTimes": [0.28, 0.31, 0.29, 0.30, 0.32, 0.27]
    }
    ```
    
    **Output:**
    - Motor baseline (median of tap times)
    - Stored in database for future sessions
    """
    try:
        db = get_db()
        
        # Compute motor baseline
        motor_baseline = compute_motor_baseline(request.tapTimes)
        
        # Store in database
        calibration_doc = {
            "userId": request.userId,
            "motorBaseline": motor_baseline,
            "calibrationDate": datetime.utcnow(),
            "tapTimes": request.tapTimes
        }
        
        db.calibrations.insert_one(calibration_doc)
        
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
    """
    Retrieve user's session history (for dashboard/visualization).
    
    **Query Parameters:**
    - `limit`: Number of recent sessions to return (default: 20)
    
    **Output:**
    - List of sessions with features and risk levels
    - Sorted by timestamp (newest first)
    """
    try:
        db = get_db()
        
        sessions = list(
            db.game_sessions.find({"userId": userId})
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        history_items = []
        for session in sessions:
            features = session.get("features", {})
            prediction = session.get("prediction", {})
            
            item = SessionHistoryItem(
                sessionId=session["sessionId"],
                timestamp=session["timestamp"].isoformat(),
                gameType=session.get("gameType", "card_matching"),
                level=session.get("level", 1),
                sac=features.get("sac", 0.0),
                ies=features.get("ies", 0.0),
                riskLevel=prediction.get("riskLevel", "UNKNOWN"),
                riskScore=prediction.get("riskScore0_100", 0.0)
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

# ============================================================================
# GET /game/stats/{userId} - Get User Statistics
# ============================================================================
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
        db = get_db()
        
        sessions = list(
            db.game_sessions.find({"userId": userId})
            .sort("timestamp", -1)
        )
        
        if not sessions:
            raise HTTPException(status_code=404, detail="No sessions found for this user")
        
        # Compute averages
        total = len(sessions)
        avg_sac = sum(s["features"]["sac"] for s in sessions) / total
        avg_ies = sum(s["features"]["ies"] for s in sessions) / total
        
        # Get most recent
        latest = sessions[0]
        current_risk = latest["prediction"]["riskLevel"]
        recent_score = latest["prediction"]["riskScore0_100"]
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

# ============================================================================
# DELETE /game/session/{sessionId} - Delete Session (Optional)
# ============================================================================
@router.delete("/session/{sessionId}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(sessionId: str):
    """
    Delete a game session (for testing/cleanup).
    """
    try:
        db = get_db()
        result = db.game_sessions.delete_one({"sessionId": sessionId})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"Deleted session {sessionId}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session")