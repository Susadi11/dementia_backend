# src/parsers/game_schemas.py
"""
Pydantic schemas for game API request/response validation
"""
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator

# ============================================================================
# Request Schemas
# ============================================================================

class GameTrial(BaseModel):
    """Single trial data from a game session"""
    rt_raw: float = Field(..., description="Raw reaction time (seconds)", gt=0)
    correct: int = Field(..., description="1 if correct, 0 otherwise", ge=0, le=1)
    error: int = Field(default=0, description="1 if error, 0 otherwise", ge=0, le=1)
    hint_used: int = Field(default=0, description="1 if hint used, 0 otherwise", ge=0, le=1)

class GameSummary(BaseModel):
    """Summary metrics (alternative to trial-level data)"""
    totalAttempts: int = Field(..., ge=1)
    correct: int = Field(..., ge=0)
    errors: int = Field(..., ge=0)
    hintsUsed: int = Field(default=0, ge=0)
    meanRtRaw: float = Field(..., gt=0, description="Mean raw reaction time")
    medianRtRaw: Optional[float] = Field(None, gt=0)

class GameSessionRequest(BaseModel):
    """
    Request body for POST /game/session
    Client can send either:
    - trials (preferred, more accurate)
    - summary (fallback, less precise)
    """
    userId: str = Field(..., min_length=1)
    sessionId: str = Field(..., min_length=1)
    gameType: str = Field(default="card_matching")
    level: int = Field(default=1, ge=1, le=10)
    timestamp: Optional[datetime] = None
    
    # One of these must be provided
    trials: Optional[List[GameTrial]] = None
    summary: Optional[GameSummary] = None
    
    @validator('trials', 'summary')
    def check_data_provided(cls, v, values):
        """Ensure at least one data format is provided"""
        if 'trials' in values and 'summary' in values:
            if values.get('trials') is None and values.get('summary') is None:
                raise ValueError("Either 'trials' or 'summary' must be provided")
        return v

class CalibrationRequest(BaseModel):
    """Request for motor baseline calibration"""
    userId: str = Field(..., min_length=1)
    tapTimes: List[float] = Field(..., min_items=5, description="At least 5 tap times")
    
    @validator('tapTimes')
    def validate_tap_times(cls, v):
        """Ensure all tap times are positive"""
        if any(t <= 0 for t in v):
            raise ValueError("All tap times must be positive")
        return v

# ============================================================================
# Response Schemas
# ============================================================================

class FeaturesResponse(BaseModel):
    """Session features (SAC, IES, etc.)"""
    accuracy: float
    errorRate: float
    rtAdjMedian: float
    sac: float
    ies: float
    variability: float

class PredictionResponse(BaseModel):
    """Risk prediction output"""
    riskProbability: Dict[str, float]
    riskLevel: str  # "LOW" | "MEDIUM" | "HIGH"
    riskScore0_100: float
    lstmDeclineScore: Optional[float] = None

class GameSessionResponse(BaseModel):
    """Complete response for game session processing"""
    sessionId: str
    userId: str
    features: FeaturesResponse
    prediction: PredictionResponse
    timestamp: str

class CalibrationResponse(BaseModel):
    """Response for calibration"""
    userId: str
    motorBaseline: float
    calibrationDate: str
    message: str = "Motor baseline calibrated successfully"

# ============================================================================
# History/Dashboard Schemas
# ============================================================================

class SessionHistoryItem(BaseModel):
    """Single session in history"""
    sessionId: str
    timestamp: str
    gameType: str
    level: int
    sac: float
    ies: float
    riskLevel: str
    riskScore: float

class SessionHistoryResponse(BaseModel):
    """List of user's session history"""
    userId: str
    totalSessions: int
    sessions: List[SessionHistoryItem]

class UserStatsResponse(BaseModel):
    """Aggregate statistics for a user"""
    userId: str
    totalSessions: int
    avgSAC: float
    avgIES: float
    currentRiskLevel: str
    recentRiskScore: float
    lastSessionDate: Optional[str] = None


# ============================================================================
# Risk Assessment Schemas
# ============================================================================

class RiskFeatures(BaseModel):
    """Features used for risk prediction"""
    mean_sac: float
    slope_sac: float
    mean_ies: float
    slope_ies: float
    mean_accuracy: float
    mean_rt: float
    mean_variability: float
    lstm_score: float

class RiskPredictionDetail(BaseModel):
    """Detailed prediction output"""
    prob_low: float
    prob_medium: float
    prob_high: float
    label: str
    risk_score_0_100: float

class RiskAssessmentResponse(BaseModel):
    """Response for /risk/predict/{user_id}"""
    user_id: str
    window_size: int
    features_used: RiskFeatures
    prediction: RiskPredictionDetail
    scale_note: str = "Risk bands align to GDS 1 / 2-3 / 4-5 (not diagnosis)"
    created_at: str

class RiskHistoryResponse(BaseModel):
    """Response for /risk/history/{user_id}"""
    user_id: str
    total_predictions: int
    history: List[RiskAssessmentResponse]
