from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from src.services.chatbot_service import get_chatbot

router = APIRouter(prefix="/chat", tags=["conversational_ai"])
logger = logging.getLogger(__name__)


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


@router.post("/voice", response_model=VoiceResponse,
             summary="Voice Chat",
             description="Send an audio message to the chatbot (requires Whisper for transcription)")
async def process_voice_chat(
    user_id: str = Form(..., description="User identifier"),
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a)"),
    session_id: Optional[str] = Form(None, description="Optional session ID"),
    max_tokens: int = Form(150, description="Maximum tokens to generate"),
    temperature: float = Form(0.7, description="Sampling temperature")
) -> VoiceResponse:
    """
    Process voice chat message.

    **Steps:**
    1. Transcribe audio to text using Whisper
    2. Send transcribed text to chatbot
    3. Return chatbot response with transcription

    **Supported formats:** wav, mp3, m4a, ogg, flac

    **Note:** Requires OpenAI Whisper or similar transcription service
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # TODO: Add Whisper transcription here
        # For now, return placeholder response
        # You can integrate Whisper API or local model later

        import tempfile
        import os

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        try:
            # Placeholder transcription (replace with actual Whisper)
            transcribed_text = "[Voice transcription - Whisper integration needed]"

            # For demo, you can manually pass text
            # In production, integrate Whisper here:
            # transcribed_text = whisper_transcribe(temp_audio_path)

            # Generate chatbot response
            chatbot = get_chatbot()
            result = chatbot.generate_response(
                user_message=transcribed_text,
                user_id=user_id,
                session_id=session_id,
                max_tokens=max_tokens,
                temperature=temperature
            )

            return VoiceResponse(
                **result,
                transcription=transcribed_text
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

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
