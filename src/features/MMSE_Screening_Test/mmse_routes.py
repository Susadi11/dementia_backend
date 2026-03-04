# src/features/mmse_screening/mmse_routes.py

from fastapi import APIRouter, UploadFile, File, Form

from .mmse_models import (
    MMSEStartResponse,
    MMSESubmitResponse,
    MMSEFinalizeResponse,
)
from .mmse_service import (
    start_mmse_assessment,
    submit_mmse_question,
    finalize_mmse_assessment,
)

router = APIRouter(prefix="/api/mmse", tags=["MMSE Screening"])


@router.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "service": "mmse_screening"}


@router.post("/start", response_model=MMSEStartResponse)
async def start_mmse(user_id: str):
    return await start_mmse_assessment(user_id)


@router.post("/submit", response_model=MMSESubmitResponse)
async def submit_mmse(
    assessment_id: str = Form(...),
    user_id: str = Form(...),
    question_type: str = Form(...),
    caregiver_is_correct: bool = Form(None),
    file: UploadFile = File(...),
):
    return await submit_mmse_question(
        assessment_id=assessment_id,
        user_id=user_id,
        question_type=question_type,
        caregiver_is_correct=caregiver_is_correct,
        file=file,
    )


@router.post("/finalize", response_model=MMSEFinalizeResponse)
async def finalize_mmse(assessment_id: str, user_id: str):
    return await finalize_mmse_assessment(assessment_id, user_id)