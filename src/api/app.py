"""
FastAPI Application for Dementia Detection

Combines conversational AI + gamified cognitive assessment for comprehensive dementia risk detection.
"""

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
import tempfile
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.features.conversational_ai.feature_extractor import FeatureExtractor
from src.models.conversational_ai.model_utils import DementiaPredictor
# Temporarily disable audio processing due to dependency issues
# from src.preprocessing.voice_processor import get_voice_processor
# from src.preprocessing.audio_models import get_db_manager
from src.routes import healthcheck, conversational_ai, reminder_routes, game_routes, risk_routes

from src.database import Database
from src.services.session_finalizer import session_finalizer

# ============================================================================
# Game Component Imports (Gamified cognitive assessment features)
# ============================================================================
# Game Component Imports
from src.models.game.model_registry import load_all_models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dementia_api')

# Initialize FastAPI app
app = FastAPI(
    title="Dementia Detection & Monitoring API",
    description="Comprehensive API combining conversational AI, gamified cognitive assessment, and intelligent reminder management for dementia risk detection",
    version="2.0.0", 
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(healthcheck.router)
app.include_router(conversational_ai.router)
app.include_router(reminder_routes.router)

# Game component routes
app.include_router(game_routes.router)
app.include_router(risk_routes.router)

# Initialize components
feature_extractor = FeatureExtractor()
predictor = DementiaPredictor()


# Pydantic models for API
class AudioData(BaseModel):
    """Audio analysis data."""
    pause_frequency: float = Field(default=0.0, ge=0, le=1, description="Frequency of pauses (0-1)")
    tremor_intensity: float = Field(default=0.0, ge=0, le=1, description="Vocal tremor intensity (0-1)")
    emotion_intensity: float = Field(default=0.0, ge=0, le=1, description="Emotional tone intensity (0-1)")
    speech_error_rate: float = Field(default=0.0, ge=0, le=1, description="Speech error rate (0-1)")
    speech_rate: float = Field(default=120.0, ge=50, le=200, description="Words per minute")


class ChatMessage(BaseModel):
    """Single chat message."""
    text: str = Field(..., description="Message text content")
    message_type: str = Field(default="text", description="Type: 'text' or 'voice'")
    audio_data: Optional[AudioData] = Field(None, description="Audio data if voice message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class AnalysisRequest(BaseModel):
    """Message analysis request."""
    text: str = Field(..., description="User message text")
    audio_data: Optional[AudioData] = Field(None, description="Audio features from voice")


class SessionAnalysisRequest(BaseModel):
    """Session analysis request."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")


class FeatureResponse(BaseModel):
    """Feature analysis response."""
    feature: str
    value: float
    percentage: float
    description: str


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    features: Dict[str, Any]
    risk_score: float
    risk_level: str
    risk_description: str
    recommendations: List[str]
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]


class AudioUploadResponse(BaseModel):
    """Response for audio upload and processing."""
    success: bool
    message: str
    session_id: str
    user_id: str
    transcript: str
    confidence: float
    duration: float
    language: str
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    error: Optional[str] = None


class AudioAnalysisResponse(BaseModel):
    """Response for audio analysis results."""
    session_id: str
    user_id: str
    transcript: str
    confidence: float
    duration: float
    language: str
    features: Dict[str, float]
    risk_score: float
    risk_level: str
    risk_description: str
    recommendations: List[str]
    timestamp: datetime


# Helper functions
def calculate_risk_level(risk_score: float) -> tuple:
    """Determine risk level and description based on score."""
    if risk_score < 0.3:
        return "low", "Low risk of dementia indicators"
    elif risk_score < 0.6:
        return "moderate", "Moderate risk - recommend further evaluation"
    else:
        return "high", "High risk indicators detected"


def extract_and_analyze(text: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract features and analyze conversation.

    Args:
        text: User message text
        audio_path: Optional path to audio file

    Returns:
        Dictionary with analysis results
    """
    try:
        # Extract features from text
        features = feature_extractor.extract_features_normalized(
            transcript_text=text,
            audio_path=audio_path
        )

        # Make prediction
        prediction, risk_score, contributions = predictor.predict(features)

        # Get risk level
        risk_level, risk_description = calculate_risk_level(risk_score)

        # Create recommendations
        recommendations = []
        if risk_level == "high":
            recommendations.append("High dementia risk detected")
            if features.get('semantic_incoherence', 0) > 0.4:
                recommendations.append("- Semantic incoherence detected in speech")
            if features.get('repeated_questions', 0) > 0.3:
                recommendations.append("- Repetitive questioning observed")
        elif risk_level == "moderate":
            recommendations.append("Moderate dementia risk indicators found")
            recommendations.append("- Recommend cognitive assessment")
        else:
            recommendations.append("✓ Low risk profile detected")
            recommendations.append("- Continue regular health monitoring")

        return {
            'features': features,
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Dementia Detection & Monitoring API",
        "version": "2.0.0",
        "docs": "/docs",
        "features": {
            "conversational_ai": "Analyze speech patterns for dementia indicators",
            "gamified_assessment": "Card-matching game with cognitive risk scoring",
            "smart_reminders": "Context-aware intelligent reminder management system"
        },
        "components": {
            "conversational": [
                "/api/analyze",
                "/api/session",
                "/api/predict",
                "/api/features",
                "/api/risk-levels"
            ],
            "detection": [
                "/api/detection/analyze-session",
                "/api/detection/weekly-risk",
                "/api/detection/session/{session_id}",
                "/api/detection/sessions/{user_id}",
                "/api/detection/active-sessions",
                "/api/detection/finalize-session/{session_id}",
                "/api/detection/run-finalization-check"
            ],
            "game": [
                "/game/session",
                "/game/calibration",
                "/game/history/{userId}",
                "/game/stats/{userId}"
            ],
            "reminders": [
                "/reminders",
                "/reminders/{reminder_id}",
                "/reminders/user/{user_id}"
            ],
            "risk": [
                "/risk/predict/{userId}",
                "/risk/history/{userId}"
            ]
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "analyzer": "operational",
                "model": "ready",
                "database": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.post("/api/analyze", tags=["Analysis"])
async def analyze_message(request: AnalysisRequest):
    """
    Analyze a single chat message.

    Accepts text and extracts dementia indicators.
    """
    logger.info(f"Analyzing message: {request.text[:50]}...")

    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content is required"
        )

    result = extract_and_analyze(request.text)

    logger.info(f"Analysis complete. Risk level: {result['risk_level']}")

    return result


@app.post("/api/session", tags=["Analysis"])
async def analyze_session(request: SessionAnalysisRequest):
    """
    Analyze a complete chat session.

    Analyzes all messages in a session.
    """
    logger.info(f"Analyzing session: {request.session_id}")

    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session must contain at least one message"
        )

    try:
        session_results = []

        # Analyze each message
        for msg in request.messages:
            if msg.text:
                result = extract_and_analyze(msg.text)
                session_results.append(result)

        # Calculate session-level metrics
        if session_results:
            avg_risk_score = sum(r['risk_score'] for r in session_results) / len(session_results)
            risk_level, risk_description = calculate_risk_level(avg_risk_score)

            # Average features
            avg_features = {}
            for feat_name in session_results[0]['features'].keys():
                values = [r['features'].get(feat_name, 0) for r in session_results]
                avg_features[feat_name] = round(sum(values) / len(values), 3)

            return {
                'session_id': request.session_id,
                'user_id': request.user_id,
                'message_count': len(request.messages),
                'average_features': avg_features,
                'average_risk_score': round(avg_risk_score, 3),
                'risk_level': risk_level,
                'risk_description': risk_description,
                'recommendations': [f"{risk_level.upper()}: {risk_description}"],
                'timestamp': datetime.now()
            }

    except Exception as e:
        logger.error(f"Session analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session analysis failed: {str(e)}"
        )


@app.get("/api/features", tags=["Information"])
async def list_features():
    """List all dementia indicator features."""
    feature_descriptions = {
        'semantic_incoherence': 'Lack of logical coherence in speech - jump between topics',
        'repeated_questions': 'Asking the same questions repeatedly',
        'self_correction': 'Frequency of self-corrections during conversation',
        'low_confidence_answer': 'Uncertainty indicators in responses',
        'hesitation_pauses': 'Number of hesitation pauses and filled pauses',
        'vocal_tremors': 'Tremors or shakiness detected in voice',
        'emotion_slip': 'Emotional intensity combined with speech errors',
        'slowed_speech': 'Reduced speech rate or speed',
        'evening_errors': 'Errors increasing during evening hours',
        'in_session_decline': 'Deterioration of performance within conversation session'
    }
    
    return {
        'total_features': 10,
        'features': [
            {
                'name': feature,
                'description': desc,
                'scale': '0-1 (0=no indicator, 1=strong indicator)'
            }
            for feature, desc in feature_descriptions.items()
        ]
    }


@app.get("/api/risk-levels", tags=["Information"])
async def get_risk_levels():
    """Get risk level definitions."""
    return {
        'low': {
            'range': '0.0 - 0.3',
            'description': 'Low risk of dementia indicators',
            'actions': ['Continue regular health monitoring', 'Maintain cognitive activities']
        },
        'moderate': {
            'range': '0.3 - 0.6',
            'description': 'Moderate risk - recommend further evaluation',
            'actions': ['Schedule cognitive assessment', 'Monitor speech patterns', 'Consider testing']
        },
        'high': {
            'range': '0.6 - 1.0',
            'description': 'High risk indicators detected',
            'actions': ['Consult neurologist', 'Comprehensive evaluation', 'Consider diagnostics']
        }
    }


@app.post("/api/predict", tags=["Prediction"])
async def predict_dementia(features: Dict[str, float]):
    """
    Make dementia risk prediction from features.

    Requires 10 feature values.
    """
    try:
        # Validate required features
        required_features = [
            'semantic_incoherence', 'repeated_questions',
            'self_correction', 'low_confidence_answers', 'hesitation_pauses',
            'vocal_tremors', 'emotion_slip', 'slowed_speech',
            'evening_errors', 'in_session_decline'
        ]

        # Normalize feature names (handle alternate naming)
        normalized_features = {}
        for feat in required_features:
            # Try exact match first
            if feat in features:
                normalized_features[feat] = features[feat]
            # Try alternate names
            elif feat.replace('_answers', '_answer') in features:
                normalized_features[feat] = features[feat.replace('_answers', '_answer')]
            else:
                normalized_features[feat] = 0.0

        # Make prediction
        prediction, risk_score, contributions = predictor.predict(normalized_features)
        risk_level, risk_description = calculate_risk_level(risk_score)

        return {
            'prediction': prediction,
            'risk_score': round(risk_score, 3),
            'risk_percentage': round(risk_score * 100, 1),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'feature_contributions': contributions,
            'recommendations': [
                f"{risk_level.upper()}: {risk_description}",
                f"Based on weighted analysis of {len([v for v in normalized_features.values() if v > 0])} detected indicators"
            ]
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ===== VOICE/AUDIO PROCESSING ENDPOINTS (Requirement 1.1) =====
# Temporarily disabled due to Python 3.14 compatibility issues with librosa/numba

# @app.post("/api/upload-audio", response_model=AudioUploadResponse, tags=["Voice Processing"])
async def upload_audio(
    file: UploadFile = File(...),
    user_id: str = "default_user",
    session_id: str = "default_session",
    language: str = "en"
):
    """Upload and process audio file with Whisper ASR transcription."""
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file selected"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Processing audio upload: {file.filename} for user {user_id}")

        voice_processor = get_voice_processor(model_size="base")
        result = voice_processor.process_audio_file(
            file_path=tmp_path,
            user_id=user_id,
            session_id=session_id,
            language=language
        )

        os.unlink(tmp_path)

        if not result['success']:
            return AudioUploadResponse(
                success=False,
                message="Audio processing failed",
                session_id=session_id,
                user_id=user_id,
                transcript="",
                confidence=0.0,
                duration=0.0,
                language=language,
                error=result['error']
            )

        db_manager = get_db_manager()
        audio_record = db_manager.save_audio_record(
            user_id=user_id,
            session_id=session_id,
            original_filename=file.filename,
            audio_path=result['audio_path'],
            transcript=result['transcript'],
            transcript_path=result['transcript_path'],
            duration=result['duration'],
            language=result['language'],
            confidence=result['confidence'],
            audio_format=Path(file.filename).suffix.lstrip('.'),
            audio_size_bytes=len(content)
        )

        logger.info(f"Audio processed and saved. Transcript: {result['transcript'][:100]}...")

        return AudioUploadResponse(
            success=True,
            message="Audio uploaded and transcribed successfully",
            session_id=session_id,
            user_id=user_id,
            transcript=result['transcript'],
            confidence=result['confidence'],
            duration=result['duration'],
            language=result['language'],
            audio_path=result['audio_path'],
            transcript_path=result['transcript_path']
        )

    except Exception as e:
        logger.error(f"Audio upload failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}"
        )


@app.post("/api/analyze-audio", response_model=AudioAnalysisResponse, tags=["Voice Processing"])
async def analyze_audio(
    file: UploadFile = File(...),
    user_id: str = "default_user",
    session_id: str = "default_session",
    language: str = "en"
):
    """Upload, transcribe, and analyze audio for dementia indicators (complete pipeline)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Analyzing audio: {file.filename} for user {user_id}")

        voice_processor = get_voice_processor(model_size="base")
        voice_result = voice_processor.process_audio_file(
            file_path=tmp_path,
            user_id=user_id,
            session_id=session_id,
            language=language
        )

        os.unlink(tmp_path)

        if not voice_result['success']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio processing failed: {voice_result['error']}"
            )

        transcript = voice_result['transcript']
        features = feature_extractor.extract_features_normalized(
            transcript_text=transcript,
            audio_path=voice_result['audio_path']
        )

        prediction, risk_score, contributions = predictor.predict(features)
        risk_level, risk_description = calculate_risk_level(risk_score)

        recommendations = []
        if risk_level == "high":
            recommendations.append("⚠️ High dementia risk detected")
            if features.get('semantic_incoherence', 0) > 0.4:
                recommendations.append("- Semantic incoherence detected in speech")
            if features.get('repeated_questions', 0) > 0.3:
                recommendations.append("- Repetitive questioning observed")
        elif risk_level == "moderate":
            recommendations.append("⚠️ Moderate dementia risk indicators found")
            recommendations.append("- Recommend cognitive assessment")
        else:
            recommendations.append("✓ Low risk profile detected")

        db_manager = get_db_manager()
        audio_record = db_manager.get_session_audio(user_id, session_id)

        if audio_record:
            db_manager.save_voice_analysis(
                audio_record_id=audio_record.id,
                user_id=user_id,
                session_id=session_id,
                features=features,
                risk_score=risk_score,
                risk_level=risk_level,
                risk_description=risk_description
            )

        logger.info(f"Audio analysis complete. Risk level: {risk_level}")

        return AudioAnalysisResponse(
            session_id=session_id,
            user_id=user_id,
            transcript=transcript,
            confidence=voice_result['confidence'],
            duration=voice_result['duration'],
            language=voice_result['language'],
            features=features,
            risk_score=round(risk_score, 3),
            risk_level=risk_level,
            risk_description=risk_description,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio analysis failed: {str(e)}"
        )


@app.get("/api/audio/{session_id}", response_model=Dict[str, Any], tags=["Voice Processing"])
async def get_audio_data(user_id: str, session_id: str):
    """Retrieve stored audio metadata and transcript for a session."""
    try:
        db_manager = get_db_manager()
        audio_record = db_manager.get_session_audio(user_id, session_id)

        if not audio_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio record not found"
            )

        logger.info(f"Retrieved audio data for session {session_id}")

        return {
            'audio_record': audio_record.to_dict(),
            'message': 'Audio data retrieved successfully'
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve audio data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audio data: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("=" * 80)
    logger.info("Dementia Detection & Monitoring API starting up...")
    logger.info("=" * 80)

    # Connect to MongoDB (Team's existing code)
    try:
        await Database.connect_to_database()
        # Create indexes for better performance
        await Database.create_indexes()
        logger.info("✓ MongoDB connected (conversational AI collections)")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        logger.warning("API will continue without database connection")

    try:
        await create_game_indexes()
        logger.info("✓ Game component indexes created")
    except Exception as e:
        logger.warning(f"Game index creation warning: {e}")
    
    try:
        load_all_models()
        logger.info("✓ Game ML models loaded")
    except Exception as e:
        logger.warning(f"Game model loading warning: {e}")

    # Start session finalizer background task
    try:
        # Run finalization check every hour
        asyncio.create_task(session_finalizer.start_background_task(interval_minutes=60))
        logger.info("✓ Session finalizer background task started (runs every 60 minutes)")
    except Exception as e:
        logger.warning(f"Session finalizer startup warning: {e}")

    logger.info("=" * 80)
    logger.info("API ready to serve requests")
    logger.info("=" * 80)


# ============================================================================
# Helper: Create Game Component Indexes
# ============================================================================
async def create_game_indexes():
    """Create indexes for game collections"""
    try:
        logger.info("Creating game component indexes...")

        # game_sessions indexes
        game_sessions = Database.get_collection("game_sessions")
        await game_sessions.create_index([("userId", 1), ("timestamp", -1)])
        await game_sessions.create_index([("userId", 1), ("sessionId", 1)], unique=True)

        # calibrations indexes
        calibrations = Database.get_collection("calibrations")
        await calibrations.create_index([("userId", 1), ("calibrationDate", -1)])
        
        # risk_predictions indexes
        risk_predictions = Database.get_collection("risk_predictions")
        await risk_predictions.create_index([("userId", 1), ("created_at", -1)])

        # alerts indexes
        alerts = Database.get_collection("alerts")
        await alerts.create_index([("userId", 1), ("timestamp", -1)])

        logger.info("✓ Game indexes created successfully")

    except Exception as e:
        logger.warning(f"Error creating game indexes: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("=" * 80)
    logger.info("Dementia Detection & Monitoring API shutting down...")
    logger.info("=" * 80)

    # Stop session finalizer background task
    try:
        session_finalizer.stop_background_task()
        logger.info("Session finalizer stopped")
    except Exception as e:
        logger.warning(f"Error stopping session finalizer: {e}")

    # Close MongoDB connection
    await Database.close_database_connection()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
