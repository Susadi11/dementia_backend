# src/features/mmse_screening/mmse_models.py

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class MMSEStartResponse(BaseModel):
    assessment_id: str = Field(..., description="MongoDB ObjectId of the created assessment")


class MMSESubmitResponse(BaseModel):
    question_score: float
    ml_prediction: str
    ml_probability: float


class MMSEFinalizeResponse(BaseModel):
    total_score: float
    ml_risk_label: str
    avg_ml_probability: float


class MMSEQuestion(BaseModel):
    question_type: str
    transcript_raw: str
    transcript_normalized: str
    ml_prediction: str
    ml_probability: float
    caregiver_is_correct: Optional[bool]
    results: Dict[str, Any]
    question_score: float
    timestamp: datetime


class MMSEAssessment(BaseModel):
    id: str
    user_id: str
    assessment_type: str = "MMSE"
    assessment_date: datetime
    questions: List[MMSEQuestion] = []
    total_score: float = 0
    ml_summary: Dict[str, Any] = {}
    status: str = "in_progress"