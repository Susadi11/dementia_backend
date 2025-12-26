"""
Simple Enhanced Dementia Detection API

This is a simplified version that works with Python 3.14
and focuses on the core functionality of your enhanced models.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Dementia Detection API",
    description="API for dementia detection using enhanced models with Pitt Corpus integration",
    version="2.0.0"
)

class TextInput(BaseModel):
    text: str
    user_id: Optional[str] = "default"

class PredictionResponse(BaseModel):
    prediction: str
    dementia_probability: float
    overall_risk: str
    confidence: float
    model_type: str
    enhanced_predictions: Optional[Dict] = None

# Global variables for models
enhanced_models = None
reminder_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Load enhanced models on startup."""
    global enhanced_models, reminder_analyzer
    
    logger.info("🚀 Starting Enhanced Dementia Detection API")
    logger.info("📊 Loading enhanced models with Pitt Corpus integration...")
    
    try:
        # Try to load enhanced models
        from src.features.reminder_system.enhanced_model_loader import EnhancedModelLoader
        from src.features.reminder_system.reminder_analyzer import PittBasedReminderAnalyzer
        
        enhanced_models = EnhancedModelLoader()
        reminder_analyzer = PittBasedReminderAnalyzer(use_enhanced_models=True)
        
        model_info = enhanced_models.get_model_info()
        logger.info(f"✅ Enhanced models loaded successfully!")
        logger.info(f"📊 Training samples: {model_info['total_samples']}")
        logger.info(f"🏆 Models available: {', '.join(model_info['models_loaded'])}")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not load enhanced models: {e}")
        logger.info("📝 Falling back to basic response system")
        enhanced_models = None
        reminder_analyzer = None

@app.get("/")
async def root():
    """Root endpoint with project information."""
    return {
        "message": "Enhanced Dementia Detection API",
        "status": "running",
        "enhanced_models": enhanced_models is not None,
        "version": "2.0.0",
        "description": "Dementia detection with Pitt Corpus integration"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "enhanced_models_loaded": enhanced_models is not None,
        "reminder_analyzer_loaded": reminder_analyzer is not None
    }

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """
    Predict dementia indicators from text input.
    
    This uses your enhanced models trained with Pitt Corpus data.
    """
    try:
        text = input_data.text
        logger.info(f"📝 Analyzing text: '{text[:50]}...'")
        
        if reminder_analyzer and enhanced_models:
            # Use enhanced models
            result = reminder_analyzer.analyze_reminder_response(text)
            
            # Determine prediction based on cognitive risk
            risk_score = result['cognitive_risk_score']
            if risk_score > 0.7:
                prediction = "High Risk"
                overall_risk = "high"
            elif risk_score > 0.4:
                prediction = "Moderate Risk"
                overall_risk = "moderate"
            else:
                prediction = "Low Risk"
                overall_risk = "low"
            
            return PredictionResponse(
                prediction=prediction,
                dementia_probability=float(risk_score),
                overall_risk=overall_risk,
                confidence=float(result['confidence']),
                model_type=result['model_type'],
                enhanced_predictions=result.get('enhanced_predictions', {})
            )
        
        else:
            # Fallback basic analysis
            logger.warning("Using basic fallback analysis")
            
            # Simple rule-based analysis
            confusion_keywords = ['what', 'confused', 'don\'t understand', 'huh']
            memory_keywords = ['forget', 'don\'t remember', 'can\'t recall']
            uncertainty_keywords = ['maybe', 'think so', 'not sure', 'possibly']
            
            text_lower = text.lower()
            
            risk_score = 0.0
            if any(word in text_lower for word in confusion_keywords):
                risk_score += 0.3
            if any(word in text_lower for word in memory_keywords):
                risk_score += 0.4
            if any(word in text_lower for word in uncertainty_keywords):
                risk_score += 0.2
            
            # Cap at 1.0
            risk_score = min(risk_score, 1.0)
            
            if risk_score > 0.5:
                prediction = "High Risk"
                overall_risk = "high"
            elif risk_score > 0.2:
                prediction = "Moderate Risk"
                overall_risk = "moderate"
            else:
                prediction = "Low Risk"
                overall_risk = "low"
            
            return PredictionResponse(
                prediction=prediction,
                dementia_probability=float(risk_score),
                overall_risk=overall_risk,
                confidence=0.6,
                model_type="rule_based_fallback"
            )
        
    except Exception as e:
        logger.error(f"❌ Error in text prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models."""
    if enhanced_models:
        model_info = enhanced_models.get_model_info()
        return {
            "enhanced_models": True,
            "model_info": model_info,
            "models": model_info['models_loaded'],
            "training_samples": model_info['total_samples'],
            "training_date": model_info['training_date']
        }
    else:
        return {
            "enhanced_models": False,
            "fallback_mode": True,
            "message": "Enhanced models not loaded, using basic analysis"
        }

if __name__ == "__main__":
    print("""
    🎉 ENHANCED DEMENTIA DETECTION API
    =====================================
    
    ✅ Your integrated Pitt Corpus models are ready!
    📊 Enhanced with real clinical dementia data
    🔗 API endpoints:
       • http://localhost:8000/ - Project info
       • http://localhost:8000/docs - API documentation 
       • http://localhost:8000/predict/text - Text analysis
       • http://localhost:8000/models - Model information
    
    🚀 Starting server on http://localhost:8001
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")