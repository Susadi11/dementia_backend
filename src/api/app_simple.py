"""
FastAPI Application for Dementia Detection (Simplified for Reminder System)

REST API focused on reminder system and text analysis.
Audio processing temporarily disabled due to Python 3.14 compatibility.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import working components
from src.features.conversational_ai.feature_extractor import FeatureExtractor
from src.models.conversational_ai.model_utils import DementiaPredictor
from src.routes import healthcheck, reminder_routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dementia_api')

# Initialize FastAPI app
app = FastAPI(
    title="Dementia Detection & Reminder System API",
    description="API for dementia risk detection and intelligent reminder management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(healthcheck.router)
app.include_router(reminder_routes.router)

# Initialize components
try:
    feature_extractor = FeatureExtractor()
    predictor = DementiaPredictor()
    logger.info("Core components initialized successfully")
except Exception as e:
    logger.warning(f"Some components failed to initialize: {e}")
    feature_extractor = None
    predictor = None


# Pydantic models for API
class AnalysisRequest(BaseModel):
    """Message analysis request."""
    text: str = Field(..., description="User message text")


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


# Helper functions
def calculate_risk_level(risk_score: float) -> tuple:
    """Determine risk level and description based on score."""
    if risk_score < 0.3:
        return "low", "Low risk of dementia indicators"
    elif risk_score < 0.6:
        return "moderate", "Moderate risk - recommend further evaluation"
    else:
        return "high", "High risk indicators detected"


def extract_and_analyze(text: str) -> Dict[str, Any]:
    """
    Extract features and analyze conversation.

    Args:
        text: User message text

    Returns:
        Dictionary with analysis results
    """
    try:
        if not feature_extractor or not predictor:
            # Return mock analysis if components not available
            return {
                'features': {},
                'risk_score': 0.0,
                'risk_level': 'unknown',
                'risk_description': 'Analysis components not available',
                'recommendations': ['System initializing - try again later'],
                'timestamp': datetime.now()
            }

        # Extract features from text
        features = feature_extractor.extract_features_normalized(
            transcript_text=text
        )

        # Make prediction
        prediction, risk_score, contributions = predictor.predict(features)

        # Get risk level
        risk_level, risk_description = calculate_risk_level(risk_score)

        # Create recommendations
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
        "message": "Dementia Detection & Reminder System API",
        "version": "1.0.0",
        "docs": "/docs",
        "features": [
            "Text-based dementia risk analysis", 
            "Context-aware smart reminders",
            "Behavioral pattern tracking",
            "Caregiver notification system"
        ],
        "note": "Audio processing temporarily disabled for Python 3.14 compatibility"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        components = {
            "reminder_system": "operational",
            "text_analyzer": "operational" if feature_extractor else "unavailable",
            "predictor": "operational" if predictor else "unavailable",
            "audio_processor": "disabled (Python 3.14 compatibility)"
        }
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": components
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_message(request: AnalysisRequest):
    """
    Analyze a single text message for dementia indicators.

    Accepts text and extracts cognitive decline indicators.
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
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not available"
            )

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Dementia Detection & Reminder System API starting up...")
    logger.info("Reminder system: ✓ Ready")
    logger.info("Text analysis: ✓ Ready" if feature_extractor else "Text analysis: ⚠️ Limited")
    logger.info("Audio processing: ⚠️ Disabled (Python 3.14 compatibility)")
    logger.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Dementia Detection & Reminder System API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )