# src/routes/risk_routes.py
"""
Risk Prediction API Routes
Endpoints:
- POST /risk/predict/{userId}
- GET /risk/history/{userId}
"""
from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional

from src.services.game_service import analyze_risk_window, get_risk_history
from src.parsers.game_schemas import RiskAssessmentResponse, RiskHistoryResponse

import logging

router = APIRouter(prefix="/risk", tags=["Risk Prediction"])
logger = logging.getLogger(__name__)

# ============================================================================
# POST /risk/predict/{userId}
# ============================================================================
@router.post("/predict/{userId}", response_model=RiskAssessmentResponse)
async def predict_user_risk(
    userId: str, 
    N: int = Query(10, description="Window size (number of past sessions)")
):
    """
    Compute dementia risk from user's last N sessions.
    
    **Workflow:**
    1. Fetch last N sessions
    2. Compute window features (trends, variability, etc.)
    3. Run risk model
    4. Store prediction
    5. Return result
    """
    try:
        logger.info(f"Predicting risk for user {userId} with window N={N}")
        result = await analyze_risk_window(userId, N)
        return result
        
    except Exception as e:
        logger.error(f"Error predicting risk: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Risk prediction failed"
        )

# ============================================================================
# GET /risk/history/{userId}
# ============================================================================
@router.get("/history/{userId}", response_model=RiskHistoryResponse)
async def get_prediction_history(userId: str):
    """
    Get history of all risk predictions made for this user.
    Useful for dashboard trend lines.
    """
    try:
        history = await get_risk_history(userId)
        
        return {
            "user_id": userId,
            "total_predictions": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error fetching risk history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch history"
        )
