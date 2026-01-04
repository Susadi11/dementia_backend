from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from src.services.chatbot_service import get_chatbot
from src.services.whisper_service import get_whisper_service
from src.services.scoring_engine import ScoringEngine
from src.services.audio_processor import audio_processor
from src.services.risk_calculator import WeeklyRiskCalculator
from src.services.session_finalizer import session_finalizer
from src.models.detection_session import (
    DetectionSessionDB,
    get_time_window_and_session
)
from src.database import Database

router = APIRouter(prefix="/chat", tags=["conversational_ai"])
logger = logging.getLogger(__name__)

# Initialize detection services
scoring_engine = ScoringEngine()
risk_calculator = WeeklyRiskCalculator()


class TextQuery(BaseModel):
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User's message text", alias="text")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    max_tokens: int = Field(150, ge=50, le=500, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    use_history: bool = Field(True, description="Use conversation history")

    class Config:
        populate_by_name = True  # Allow both 'message' and 'text' field names


class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    user_id: str = Field(..., description="User identifier")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    safety_warnings: Optional[list] = Field(None, description="Safety warnings if any")

    # 12-Parameter Detection (Automatic)
    detection: Optional[Dict[str, Any]] = Field(None, description="Automatic dementia detection scores (12 parameters)")


@router.post("/text", response_model=ChatResponse,
             summary="Text Chat",
             description="Send a text message to the dementia care chatbot")
async def process_text_chat(request: TextQuery) -> ChatResponse:
    """
    Process text-based chat message using fine-tuned LLaMA 3.2 1B model.

    The chatbot is trained on DailyDialog dataset and optimized for:
    - Empathetic conversations with elderly users
    - Memory-related concerns
    - Patient and clear responses

    **Example Request:**
    ```json
    {
        "user_id": "elderly-user-001",
        "message": "I can't remember where I put my glasses",
        "session_id": "session_123",
        "max_tokens": 150,
        "temperature": 0.7
    }
    ```
    """
    try:
        chatbot = get_chatbot()

        # Use 'message' if present, otherwise fall back to 'text'
        user_message = request.message

        result = chatbot.generate_response(
            user_message=user_message,
            user_id=request.user_id,
            session_id=request.session_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_history=request.use_history
        )

        # ==================== AUTOMATIC 12-PARAMETER DETECTION ====================
        detection_result = None
        try:
            timestamp = datetime.now()
            time_window, session_number = get_time_window_and_session(timestamp)
            date_str = timestamp.strftime("%Y-%m-%d")
            detection_session_id = f"{request.user_id}_{date_str}_{time_window}"

            db = Database.db

            # Get or create detection session
            session = await DetectionSessionDB.get_or_create_session(
                db=db,
                session_id=detection_session_id,
                user_id=request.user_id,
                date=date_str,
                time_window=time_window,
                session_number=session_number,
                timestamp=timestamp
            )

            # Only run detection if session is active
            if session.get("status") == "active":
                # Append message to session
                message_data = {
                    "timestamp": timestamp,
                    "text": user_message,
                    "audio_features": None
                }

                await DetectionSessionDB.append_message_to_session(
                    db=db,
                    session_id=detection_session_id,
                    message_data=message_data,
                    text=user_message,
                    timestamp=timestamp
                )

                # Get conversation context
                session = await DetectionSessionDB.get_session_by_id(db, detection_session_id)
                conversation_context = session.get("conversation_context", [])

                # Run 12-parameter detection
                analysis_result = scoring_engine.analyze_session(
                    text=user_message,
                    audio_features=None,
                    timestamp=timestamp,
                    conversation_context=conversation_context
                )

                scores = analysis_result["scores"]
                session_raw_score = analysis_result["session_raw_score"]

                # Update session with new scores
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

                await DetectionSessionDB.update_session(db, detection_session_id, update_data)

                # Prepare detection result for response
                detection_result = {
                    "detection_session_id": detection_session_id,
                    "time_window": time_window,
                    "session_number": session_number,
                    "message_count": len(conversation_context),
                    "parameters": scores,
                    "session_raw_score": session_raw_score,
                    "max_possible_score": 36,
                    "analysis_details": analysis_result["analysis_details"]
                }

                logger.info(f"✓ Detection completed: {detection_session_id}, score: {session_raw_score}/36")

        except Exception as e:
            logger.warning(f"Detection failed (non-critical): {str(e)}")
            # Detection failure doesn't break the chat

        # ==================== END DETECTION ====================

        # Add detection to result
        result["detection"] = detection_result

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error processing text chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


class VoiceResponse(BaseModel):
    response: str = Field(..., description="Chatbot's response")
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User identifier")
    timestamp: str = Field(..., description="Response timestamp")
    transcription: str = Field(..., description="Transcribed text from audio")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    safety_warnings: Optional[list] = Field(None, description="Safety warnings if any")

    # 12-Parameter Detection (Automatic)
    detection: Optional[Dict[str, Any]] = Field(None, description="Automatic dementia detection scores (12 parameters)")
    audio_features: Optional[Dict[str, float]] = Field(None, description="Extracted audio features (P5, P6, P8)")


@router.post("/voice", response_model=VoiceResponse,
             summary="Voice Chat",
             description="Send an audio message to the chatbot with Whisper transcription")
async def process_voice_chat(
    user_id: str = Form(..., description="User identifier"),
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, ogg, flac)"),
    session_id: Optional[str] = Form(None, description="Optional session ID"),
    max_tokens: int = Form(150, description="Maximum tokens to generate"),
    temperature: float = Form(0.7, description="Sampling temperature"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es') - Leave empty for auto-detection", example="en")
) -> VoiceResponse:
    """
    Process voice chat message with local Whisper transcription.

    **Flow:**
    1. **Whisper** → Transcribe audio to text (local, free)
    2. **spaCy + NLP** → Analyze intent, emotion, greeting detection
    3. **Prompt Builder** → Build context-aware prompts
    4. **LLaMA** → Generate empathetic text response
    5. **Return** → Text response with transcription

    **Supported formats:** wav, mp3, m4a, ogg, flac

    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/chat/voice" \\
      -F "user_id=user_001" \\
      -F "file=@my_audio.wav"
    ```
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        logger.info(f"Processing voice message from user: {user_id}")

        import tempfile
        import os
        from pathlib import Path

        # Get file extension for proper handling
        file_ext = Path(file.filename).suffix or ".wav"

        # Log file size for debugging
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        logger.info(f"Processing voice message... Size: {file_size} bytes")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        try:
            # Step 0: Extract audio features for P5, P6, P8 (pause, tremor, speech rate)
            logger.info("🎵 Step 0: Extracting audio features (P5, P6, P8)...")
            audio_features = None
            try:
                audio_features = audio_processor.extract_features_from_file(temp_audio_path)
                logger.info(f"✅ Audio features: {audio_features}")
            except Exception as e:
                logger.warning(f"Audio feature extraction failed (non-critical): {str(e)}")
                # If audio processing fails, use None - detection will use text-only

            # Step 1: Transcribe audio using local Whisper
            logger.info("📝 Step 1: Transcribing audio with Whisper...")
            whisper_service = get_whisper_service()

            # Sanitize language parameter - Swagger sends "string" as placeholder
            # Only pass language if it's a valid 2-letter code, otherwise let Whisper auto-detect
            valid_language = None
            if language and language.lower() != "string" and len(language) == 2:
                valid_language = language.lower()

            transcription_result = whisper_service.transcribe(
                audio_path=temp_audio_path,
                language=valid_language
            )

            transcribed_text = transcription_result["text"]
            logger.info(f"✅ Transcription: '{transcribed_text[:100]}...'")

            if not transcribed_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Could not transcribe audio. Please ensure clear speech."
                )

            # Step 2-4: NLP analysis + Prompt building + LLaMA response
            # (This happens automatically inside the chatbot service)
            logger.info("🧠 Step 2-4: NLP analysis → Prompt building → LLaMA generation...")
            chatbot = get_chatbot()
            result = chatbot.generate_response(
                user_message=transcribed_text,
                user_id=user_id,
                session_id=session_id,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Add transcription info to metadata
            if result.get("metadata"):
                result["metadata"]["transcription"] = {
                    "text": transcribed_text,
                    "language": transcription_result["language"],
                    "confidence": transcription_result["confidence"],
                    "duration": transcription_result["duration"]
                }

            logger.info(f"✅ Response generated: '{result['response'][:50]}...'")

            # ==================== AUTOMATIC 12-PARAMETER DETECTION ====================
            detection_result = None
            try:
                timestamp = datetime.now()
                time_window, session_number = get_time_window_and_session(timestamp)
                date_str = timestamp.strftime("%Y-%m-%d")
                detection_session_id = f"{user_id}_{date_str}_{time_window}"

                db = Database.db

                # Get or create detection session
                session = await DetectionSessionDB.get_or_create_session(
                    db=db,
                    session_id=detection_session_id,
                    user_id=user_id,
                    date=date_str,
                    time_window=time_window,
                    session_number=session_number,
                    timestamp=timestamp
                )

                # Only run detection if session is active
                if session.get("status") == "active":
                    # Append message to session
                    message_data = {
                        "timestamp": timestamp,
                        "text": transcribed_text,
                        "audio_features": audio_features
                    }

                    await DetectionSessionDB.append_message_to_session(
                        db=db,
                        session_id=detection_session_id,
                        message_data=message_data,
                        text=transcribed_text,
                        timestamp=timestamp
                    )

                    # Get conversation context
                    session = await DetectionSessionDB.get_session_by_id(db, detection_session_id)
                    conversation_context = session.get("conversation_context", [])

                    # Run 12-parameter detection (with audio features!)
                    analysis_result = scoring_engine.analyze_session(
                        text=transcribed_text,
                        audio_features=audio_features,
                        timestamp=timestamp,
                        conversation_context=conversation_context
                    )

                    scores = analysis_result["scores"]
                    session_raw_score = analysis_result["session_raw_score"]

                    # Update session with new scores
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

                    await DetectionSessionDB.update_session(db, detection_session_id, update_data)

                    # Prepare detection result for response
                    detection_result = {
                        "detection_session_id": detection_session_id,
                        "time_window": time_window,
                        "session_number": session_number,
                        "message_count": len(conversation_context),
                        "parameters": scores,
                        "session_raw_score": session_raw_score,
                        "max_possible_score": 36,
                        "analysis_details": analysis_result["analysis_details"]
                    }

                    logger.info(f"✓ Voice detection completed: {detection_session_id}, score: {session_raw_score}/36")

            except Exception as e:
                logger.warning(f"Detection failed (non-critical): {str(e)}")
                # Detection failure doesn't break the chat

            # ==================== END DETECTION ====================

            # Add detection and audio features to result
            result["detection"] = detection_result
            result["audio_features"] = audio_features

            return VoiceResponse(
                **result,
                transcription=transcribed_text
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                logger.debug(f"Cleaned up temp file: {temp_audio_path}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice chat: {str(e)}"
        )


@router.get("/sessions/{session_id}",
            summary="Get Session History",
            description="Retrieve conversation history for a session")
async def get_session_history(session_id: str) -> Dict[str, Any]:
    """
    Get conversation history for a specific session.

    Returns all messages exchanged in the session.
    """
    try:
        chatbot = get_chatbot()
        history = chatbot.get_session_history(session_id)

        if history is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        # Format history into user/assistant pairs
        messages = []
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                messages.append({
                    "user": history[i],
                    "assistant": history[i + 1]
                })

        return {
            "session_id": session_id,
            "message_count": len(history),
            "exchange_count": len(messages),
            "messages": messages
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving session history")


@router.delete("/sessions/{session_id}",
               summary="Clear Session",
               description="Delete conversation history for a session")
async def clear_session(session_id: str) -> Dict[str, str]:
    """
    Clear conversation history for a specific session.

    Useful for:
    - Starting fresh conversations
    - Privacy/data management
    - Testing
    """
    try:
        chatbot = get_chatbot()
        success = chatbot.clear_session(session_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )

        return {
            "message": f"Session {session_id} cleared successfully",
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail="Error clearing session")


@router.get("/health",
            summary="Chatbot Health Check",
            description="Check if chatbot model is loaded and ready")
async def chatbot_health() -> Dict[str, Any]:
    """
    Health check endpoint for chatbot service.

    Returns model status and configuration.
    """
    try:
        chatbot = get_chatbot()

        return {
            "status": "healthy",
            "model_loaded": chatbot.model is not None,
            "tokenizer_loaded": chatbot.tokenizer is not None,
            "device": chatbot.device,
            "base_model": chatbot.base_model_name,
            "lora_adapter": str(chatbot.lora_adapter_path),
            "active_sessions": len(chatbot.conversation_history)
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ==================== DETECTION & ANALYTICS ENDPOINTS ====================

@router.get("/weekly-risk",
            summary="Get Weekly Risk",
            description="Calculate weekly dementia risk for a user")
async def get_weekly_risk(user_id: str, week_start: str) -> Dict[str, Any]:
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
                status_code=400,
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
                status_code=404,
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
            status_code=500,
            detail=f"Weekly risk calculation failed: {str(e)}"
        )


@router.get("/detection-session/{session_id}",
            summary="Get Detection Session",
            description="Retrieve a specific detection session by ID")
async def get_detection_session(session_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific detection session by ID.

    Args:
        session_id: Unique session identifier

    Returns:
        Session data with all 12 parameters
    """
    try:
        db = Database.db
        session = await DetectionSessionDB.get_session_by_id(db, session_id)

        if not session:
            raise HTTPException(
                status_code=404,
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
            status_code=500,
            detail=f"Session retrieval failed: {str(e)}"
        )


@router.get("/detection-sessions/{user_id}",
            summary="Get User Detection Sessions",
            description="Retrieve all detection sessions for a user")
async def get_user_detection_sessions(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve all detection sessions for a user (optionally filtered by date range).

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
                    status_code=400,
                    detail="Invalid start_date format. Use YYYY-MM-DD"
                )

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
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
            status_code=500,
            detail=f"Sessions retrieval failed: {str(e)}"
        )


@router.get("/active-detection-sessions",
            summary="Get Active Detection Sessions",
            description="Get all active (non-finalized) detection sessions")
async def get_active_detection_sessions(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all active (non-finalized) detection sessions.

    Query Parameters:
        user_id: Optional user ID filter

    Returns:
        List of active sessions
    """
    try:
        db = Database.db
        collection = db["detection_sessions"]

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
            status_code=500,
            detail=f"Failed to retrieve active sessions: {str(e)}"
        )


@router.post("/finalize-session/{session_id}",
             summary="Manually Finalize Session",
             description="Force finalization of a specific session")
async def manually_finalize_session(session_id: str) -> Dict[str, Any]:
    """
    Manually finalize a specific detection session.

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
                status_code=404,
                detail=f"Session not found: {session_id}"
            )

        # Check if already finalized
        if session.get("status") == "finalized":
            raise HTTPException(
                status_code=400,
                detail="Session already finalized"
            )

        # Finalize session
        success = await session_finalizer.finalize_session_with_scores(db, session)

        if not success:
            raise HTTPException(
                status_code=500,
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
            status_code=500,
            detail=f"Manual finalization failed: {str(e)}"
        )


@router.post("/run-finalization-check",
             summary="Run Finalization Check",
             description="Manually trigger finalization check for all active sessions")
async def run_finalization_check() -> Dict[str, Any]:
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
            status_code=500,
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
