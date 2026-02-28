"""
Detection Routes

Endpoints for 12-parameter detection and weekly risk calculation.
"""

from fastapi import APIRouter, HTTPException, status, File, UploadFile
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import uuid

from src.database import Database
from src.services.chatbot import ScoringEngine, WeeklyRiskCalculator, session_finalizer, audio_processor
from src.models.detection_session import (
    DetectionSessionModel,
    DetectionSessionDB,
    get_time_window_and_session
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detection", tags=["Detection"])

# Initialize services
scoring_engine = ScoringEngine()
risk_calculator = WeeklyRiskCalculator()


# ==================== REQUEST MODELS ====================

class AudioFeaturesRequest(BaseModel):
    """Audio features from voice analysis"""
    pause_frequency: Optional[float] = Field(default=0.0, ge=0, le=1)
    tremor_intensity: Optional[float] = Field(default=0.0, ge=0, le=1)
    emotion_intensity: Optional[float] = Field(default=0.0, ge=0, le=1)
    speech_rate: Optional[float] = Field(default=120.0, ge=50, le=200)


class SessionAnalysisRequest(BaseModel):
    """Request for single session analysis"""
    user_id: str = Field(default="demo_user", description="User ID")
    text: str = Field(..., min_length=1, description="User chat text")
    audio_features: Optional[AudioFeaturesRequest] = Field(None, description="Audio features")
    timestamp: Optional[datetime] = Field(default=None, description="Session timestamp")
    conversation_context: Optional[List[str]] = Field(
        default=None,
        description="Previous messages in conversation"
    )


class WeeklyRiskRequest(BaseModel):
    """Request for weekly risk calculation"""
    user_id: str = Field(..., description="User ID")
    week_start: datetime = Field(..., description="Week start date (YYYY-MM-DD)")


# ==================== ENDPOINTS ====================

@router.post("/analyze-session", response_model=Dict[str, Any])
async def analyze_session(request: SessionAnalysisRequest):
    """
    Analyze a chat message and accumulate in session.

    Process:
    1. Determine time window from timestamp
    2. Get or create session for this time window
    3. Append message to session
    4. Re-calculate scores based on all messages
    5. Update session in database (remains active)
    6. Return current scores

    Session Accumulation Logic:
    - Multiple chat periods in same time window = same session
    - Session stays "active" until finalized
    - Finalization happens via background job when time window ends

    Returns:
        Current session scores (will be updated as more messages arrive)
    """
    try:
        # Use current time if not provided
        timestamp = request.timestamp or datetime.now()

        # Get time window and session number
        time_window, session_number = get_time_window_and_session(timestamp)
        date_str = timestamp.strftime("%Y-%m-%d")
        session_id = f"{request.user_id}_{date_str}_{time_window}"

        db = Database.db

        # Get or create session
        session = await DetectionSessionDB.get_or_create_session(
            db=db,
            session_id=session_id,
            user_id=request.user_id,
            date=date_str,
            time_window=time_window,
            session_number=session_number,
            timestamp=timestamp
        )

        # Check if session is already finalized
        if session.get("status") == "finalized":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Session already finalized. Time window {time_window} has ended."
            )

        # Convert audio features to dict
        audio_features_dict = None
        if request.audio_features:
            audio_features_dict = request.audio_features.dict()

        # Append message to session
        message_data = {
            "timestamp": timestamp,
            "text": request.text,
            "audio_features": audio_features_dict
        }

        await DetectionSessionDB.append_message_to_session(
            db=db,
            session_id=session_id,
            message_data=message_data,
            text=request.text,
            timestamp=timestamp
        )

        # Get updated session with all messages
        session = await DetectionSessionDB.get_session_by_id(db, session_id)

        # Get all conversation context from session
        conversation_context = session.get("conversation_context", [])

        # Re-calculate scores based on ALL messages in session
        analysis_result = scoring_engine.analyze_session(
            text=request.text,
            audio_features=audio_features_dict,
            timestamp=timestamp,
            conversation_context=conversation_context
        )

        scores = analysis_result["scores"]
        session_raw_score = analysis_result["session_raw_score"]

        # Update session with new scores (still active)
        update_data = {
            "p1_semantic_incoherence": scores["p1_semantic_incoherence"],
            "p2_repeated_questions": scores["p2_repeated_questions"],
            "p3_self_correction": scores["p3_self_correction"],
            "p4_low_confidence": scores["p4_low_confidence"],
            "p5_hesitation_pauses": scores["p5_hesitation_pauses"],
            "p6_vocal_tremors": scores["p6_vocal_tremors"],
            "p7_emotion_slip": scores["p7_emotion_slip"],
            "p8_slowed_speech": scores["p8_slowed_speech"],
            "p9_evening_errors": scores["p9_evening_errors"],
            "p10_in_session_decline": scores["p10_in_session_decline"],
            "p11_memory_recall_failure": scores["p11_memory_recall_failure"],
            "p12_topic_maintenance": scores["p12_topic_maintenance"],
            "session_raw_score": session_raw_score
        }

        await DetectionSessionDB.update_session(db, session_id, update_data)

        # Calculate message count
        message_count = len(conversation_context)

        logger.info(
            f"Message added to session: {session_id}, "
            f"messages: {message_count}, current score: {session_raw_score}/36"
        )

        # Return response
        return {
            "success": True,
            "session_id": session_id,
            "user_id": request.user_id,
            "timestamp": timestamp,
            "date": date_str,
            "time_window": time_window,
            "session_number": session_number,
            "status": "active",

            # Session info
            "message_count": message_count,
            "session_start": session.get("session_start"),
            "last_message_at": timestamp,

            # Current scores (will update as more messages arrive)
            "parameters": scores,
            "session_raw_score": session_raw_score,
            "max_possible_score": 36,

            # Analysis details
            "analysis_details": analysis_result["analysis_details"],

            # Important notes
            "note": f"Message added to active session. Scores will update with each message. Session will finalize when {time_window} window ends."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session analysis failed: {str(e)}"
        )


@router.post("/process-audio", response_model=Dict[str, Any])
async def process_audio(audio_file: UploadFile = File(...)):
    """
    Process audio file and extract features for P5, P6, P8.

    This endpoint:
    1. Receives audio file (WAV, MP3, etc.)
    2. Extracts audio features using librosa:
       - P5: Pause frequency (hesitation pauses)
       - P6: Tremor intensity (vocal tremors)
       - P8: Speech rate (words per minute)
       - P7: Emotion intensity (for emotion detection)
    3. Returns features that can be used in analyze-session

    Use Case:
        Client sends audio → Get features → Send to analyze-session with text

    Returns:
        Audio features dictionary ready for analyze-session endpoint
    """
    try:
        # Read audio file bytes
        audio_bytes = await audio_file.read()

        # Extract features using audio processor
        features = audio_processor.extract_features_from_bytes(audio_bytes)

        logger.info(f"Audio features extracted: {features}")

        return {
            "success": True,
            "filename": audio_file.filename,
            "audio_features": features,
            "message": "Audio features extracted successfully. Use these in analyze-session endpoint.",
            "usage_example": {
                "endpoint": "/api/detection/analyze-session",
                "audio_features_field": features
            }
        }

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}"
        )


@router.get("/weekly-risk", response_model=Dict[str, Any])
async def get_weekly_risk(user_id: str, week_start: str):
    """
    Calculate weekly dementia risk for a user.

    Process:
    1. Retrieve all sessions for the week
    2. Calculate weekly average score
    3. Normalize to 0-100
    4. Apply trend vs previous week
    5. Calculate final risk (0-100)

    Query Parameters:
        user_id: User ID
        week_start: Week start date (YYYY-MM-DD format)

    Returns:
        Weekly risk metrics with breakdown
    """
    try:
        # Parse week start date
        try:
            week_start_dt = datetime.strptime(week_start, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )

        # Calculate weekly risk
        db = Database.db
        risk_result = await risk_calculator.calculate_weekly_risk(
            db=db,
            user_id=user_id,
            week_start=week_start_dt
        )

        # Check if error (no sessions)
        if "error" in risk_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=risk_result["error"]
            )

        logger.info(
            f"Weekly risk retrieved for {user_id}: "
            f"{risk_result['final_weekly_risk']:.2f} ({risk_result['risk_level']})"
        )

        return {
            "success": True,
            **risk_result,
            "interpretation": _get_risk_interpretation(risk_result["risk_level"])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weekly risk calculation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weekly risk calculation failed: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """
    Retrieve a specific session by ID.

    Args:
        session_id: Unique session identifier

    Returns:
        Session data with all parameters
    """
    try:
        db = Database.db
        session = await DetectionSessionDB.get_session_by_id(db, session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Remove MongoDB _id field
        if "_id" in session:
            session["_id"] = str(session["_id"])

        return {
            "success": True,
            "session": session
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session retrieval failed: {str(e)}"
        )


@router.get("/sessions/{user_id}", response_model=Dict[str, Any])
async def get_user_sessions(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Retrieve all sessions for a user (optionally filtered by date range).

    Query Parameters:
        user_id: User ID
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        List of sessions
    """
    try:
        # Parse dates if provided
        start_dt = None
        end_dt = None

        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid start_date format. Use YYYY-MM-DD"
                )

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid end_date format. Use YYYY-MM-DD"
                )

        # Retrieve sessions
        db = Database.db
        sessions = await DetectionSessionDB.get_sessions_by_user(
            db=db,
            user_id=user_id,
            start_date=start_dt,
            end_date=end_dt
        )

        # Clean MongoDB _id fields
        for session in sessions:
            if "_id" in session:
                session["_id"] = str(session["_id"])

        return {
            "success": True,
            "user_id": user_id,
            "sessions_count": len(sessions),
            "start_date": start_date,
            "end_date": end_date,
            "sessions": sessions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sessions retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sessions retrieval failed: {str(e)}"
        )


# ==================== UTILITY ENDPOINTS ====================

@router.get("/active-sessions", response_model=Dict[str, Any])
async def get_active_sessions(user_id: Optional[str] = None):
    """
    Get all active (non-finalized) sessions.

    Query Parameters:
        user_id: Optional user ID filter

    Returns:
        List of active sessions
    """
    try:
        db = Database.db
        collection = db["chat_detection_sessions"]

        # Build query
        query = {"status": "active"}
        if user_id:
            query["user_id"] = user_id

        cursor = collection.find(query).sort("last_message_at", -1)
        sessions = await cursor.to_list(length=None)

        # Clean MongoDB _id fields
        for session in sessions:
            if "_id" in session:
                session["_id"] = str(session["_id"])

        logger.info(f"Retrieved {len(sessions)} active sessions")

        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Failed to retrieve active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve active sessions: {str(e)}"
        )


@router.post("/finalize-session/{session_id}", response_model=Dict[str, Any])
async def manually_finalize_session(session_id: str):
    """
    Manually finalize a specific session.

    Use this to force finalization before time window ends.

    Args:
        session_id: Session ID to finalize

    Returns:
        Finalized session data
    """
    try:
        db = Database.db

        # Get session
        session = await DetectionSessionDB.get_session_by_id(db, session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Check if already finalized
        if session.get("status") == "finalized":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Session already finalized"
            )

        # Finalize session
        success = await session_finalizer.finalize_session_with_scores(db, session)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to finalize session"
            )

        # Get updated session
        updated_session = await DetectionSessionDB.get_session_by_id(db, session_id)

        if "_id" in updated_session:
            updated_session["_id"] = str(updated_session["_id"])

        logger.info(f"Session manually finalized: {session_id}")

        return {
            "success": True,
            "message": "Session finalized successfully",
            "session": updated_session
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual finalization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Manual finalization failed: {str(e)}"
        )


@router.post("/run-finalization-check", response_model=Dict[str, Any])
async def run_finalization_check():
    """
    Manually trigger finalization check for all active sessions.

    This runs the same logic as the background job.
    Useful for testing or immediate finalization.

    Returns:
        Summary of finalization check
    """
    try:
        logger.info("Manual finalization check triggered")

        await session_finalizer.run_finalization_check()

        return {
            "success": True,
            "message": "Finalization check completed",
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Finalization check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Finalization check failed: {str(e)}"
        )


# ==================== HELPER FUNCTIONS ====================

def _get_risk_interpretation(risk_level: str) -> Dict[str, Any]:
    """Get interpretation and recommendations for risk level"""
    interpretations = {
        "Normal": {
            "description": "Low risk of dementia indicators",
            "recommendations": [
                "Continue regular health monitoring",
                "Maintain cognitive activities",
                "Keep using the app regularly"
            ],
            "color": "green"
        },
        "Mild": {
            "description": "Mild risk indicators detected",
            "recommendations": [
                "Monitor speech patterns more closely",
                "Consider lifestyle adjustments",
                "Schedule check-in with healthcare provider"
            ],
            "color": "yellow"
        },
        "Moderate": {
            "description": "Moderate risk - recommend further evaluation",
            "recommendations": [
                "Schedule cognitive assessment",
                "Consult with healthcare provider",
                "Increase monitoring frequency"
            ],
            "color": "orange"
        },
        "High": {
            "description": "High risk indicators detected",
            "recommendations": [
                "Consult neurologist soon",
                "Comprehensive cognitive evaluation recommended",
                "Consider diagnostic testing"
            ],
            "color": "red"
        },
        "Critical": {
            "description": "Critical risk level - immediate attention needed",
            "recommendations": [
                "Seek immediate medical consultation",
                "Comprehensive neurological evaluation urgent",
                "Alert caregivers and family"
            ],
            "color": "darkred"
        }
    }

    return interpretations.get(risk_level, interpretations["Normal"])
