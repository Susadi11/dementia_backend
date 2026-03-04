# src/features/mmse_screening/mmse_service.py

from datetime import datetime

from fastapi import HTTPException, UploadFile

from src.services.mmse_question_service import process_mmse_question
from src.utils.text_normalizer import normalize

from .mmse_audio_service import process_audio_upload
from .mmse_db_service import db_service
from .mmse_ml_service import late_fusion_predict
from .mmse_models import (
    MMSEStartResponse,
    MMSESubmitResponse,
    MMSEFinalizeResponse,
)


async def start_mmse_assessment(user_id: str) -> MMSEStartResponse:
    user = await db_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    assessment_id = await db_service.create_assessment(user_id)
    return MMSEStartResponse(assessment_id=assessment_id)


async def submit_mmse_question(
    assessment_id: str,
    user_id: str,
    question_type: str,
    caregiver_is_correct: bool,
    file: UploadFile,
) -> MMSESubmitResponse:
    # Validate assessment id & user link by trying to fetch it
    existing = await db_service.get_assessment(assessment_id, user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Assessment not found")

    # 1. Audio pipeline
    audio_features, raw_transcript = await process_audio_upload(file)

    # 2. Text normalization for MMSE scoring
    normalized_transcript = normalize(raw_transcript)

    # 3. ML prediction (late fusion)
    ml_label, ml_prob, _ = late_fusion_predict(audio_features, raw_transcript)

    # 4. Rule-based MMSE scoring
    results, question_score = process_mmse_question(
        question_type=question_type,
        transcript=normalized_transcript,
        ml_prediction=ml_label,
        caregiver_is_correct=caregiver_is_correct,
    )

    # 5. Persist question
    question_doc = {
        "question_type": question_type,
        "transcript_raw": raw_transcript,
        "transcript_normalized": normalized_transcript,
        "ml_prediction": ml_label,
        "ml_probability": ml_prob,
        "caregiver_is_correct": caregiver_is_correct,
        "results": results,
        "question_score": float(question_score),
        "timestamp": datetime.utcnow(),
    }

    await db_service.add_question(
        assessment_id=assessment_id,
        user_id=user_id,
        question_doc=question_doc,
    )

    return MMSESubmitResponse(
        question_score=float(question_score),
        ml_prediction=ml_label,
        ml_probability=ml_prob,
    )


async def finalize_mmse_assessment(
    assessment_id: str,
    user_id: str,
) -> MMSEFinalizeResponse:
    assessment = await db_service.get_assessment(assessment_id, user_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    questions = assessment.get("questions", [])
    if not questions:
        raise HTTPException(status_code=400, detail="No questions recorded for this assessment")

    total_score = float(sum(q.get("question_score", 0) for q in questions))
    avg_ml_prob = float(
        sum(q.get("ml_probability", 0) for q in questions) / len(questions)
    )

    ml_risk_label = "Dementia" if avg_ml_prob >= 0.60 else "Control"

    await db_service.finalize_assessment(
        assessment_id=assessment_id,
        total_score=total_score,
        avg_ml_prob=avg_ml_prob,
        ml_label=ml_risk_label,
    )

    return MMSEFinalizeResponse(
        total_score=total_score,
        ml_risk_label=ml_risk_label,
        avg_ml_probability=avg_ml_prob,
    )