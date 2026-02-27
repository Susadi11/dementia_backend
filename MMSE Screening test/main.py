from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId
import numpy as np
import librosa
import joblib
import whisper
import tempfile
import os
import subprocess
import uuid

from services.mmse_question_service import process_mmse_question
from utils.text_normalizer import normalize 


# LOAD ENV VARIABLES
load_dotenv()

MONGO_URL = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DB_NAME")

if not MONGO_URL:
    raise ValueError("MONGO_URL not found in environment variables")


# FASTAPI INIT
app = FastAPI(title="AI Dementia Detection API")


# DATABASE CONNECTION
client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]

users_collection = db["users"]
assessments_collection = db["assessments"]


# LOAD MODELS
MODEL_PATH = os.getenv("MODEL_PATH", "models/")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

audio_model = joblib.load(f"{MODEL_PATH}best_audio_model.pkl")
audio_scaler = joblib.load(f"{MODEL_PATH}audio_scaler.pkl")

text_model = joblib.load(f"{MODEL_PATH}best_text_model.pkl")
text_tfidf = joblib.load(f"{MODEL_PATH}text_tfidf.pkl")
text_pca = joblib.load(f"{MODEL_PATH}text_pca.pkl")

whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

# AUDIO UTILITIES
def convert_to_wav(input_path: str) -> str:
    out_path = f"{input_path}_{uuid.uuid4().hex}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return out_path


def extract_audio_features(path, sr_target=16000):
    y, sr = librosa.load(path, sr=sr_target, mono=True)
    y = librosa.util.normalize(y)

    if len(y) < sr:
        raise HTTPException(status_code=400, detail="Audio too short")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr
    )

    features = np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta2.mean(axis=1),
        chroma.mean(axis=1),
        contrast.mean(axis=1),
        tonnetz.mean(axis=1),
    ])

    return features.reshape(1, -1)


def transcribe_audio(path):
    result = whisper_model.transcribe(
        path,
        language="en",
        fp16=False
    )
    raw_text = result.get("text", "").strip().lower()
    return raw_text


def late_fusion_predict(audio_feat, transcript):
    audio_feat = audio_scaler.transform(audio_feat)
    audio_prob = audio_model.predict_proba(audio_feat)[0][1]

    if len(transcript.split()) < 5:
        fused = audio_prob
    else:
        tfidf_vec = text_tfidf.transform([transcript]).toarray()
        text_vec = text_pca.transform(tfidf_vec)
        text_prob = text_model.predict_proba(text_vec)[0][1]
        fused = 0.6 * audio_prob + 0.4 * text_prob

    label = "Dementia" if fused >= 0.60 else "Control"

    return label, fused


# ROOT
@app.get("/")
async def root():
    return {"message": "AI Dementia Detection API is running."}


# START MMSE TEST
@app.post("/mmse/start")
async def start_mmse(user_id: str):

    user = await users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    assessment = {
        "user_id": user_id,
        "assessment_type": "MMSE",
        "assessment_date": datetime.utcnow(),
        "questions": [],
        "total_score": 0,
        "ml_summary": {},
        "status": "in_progress"
    }

    result = await assessments_collection.insert_one(assessment)

    return {
        "assessment_id": str(result.inserted_id)
    }


# SUBMIT MMSE QUESTION
@app.post("/mmse/submit")
async def submit_mmse_question(
    assessment_id: str = Form(...),
    user_id: str = Form(...),
    question_type: str = Form(...),
    caregiver_is_correct: bool = Form(None),
    file: UploadFile = File(...)
):

    if not ObjectId.is_valid(assessment_id):
        raise HTTPException(status_code=400, detail="Invalid assessment ID")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    try:
        wav_path = convert_to_wav(input_path)

        audio_feat = extract_audio_features(wav_path)
        raw_transcript = transcribe_audio(wav_path)
        normalized_transcript = normalize(raw_transcript)  #for MMSE scoring

        ml_label, ml_prob = late_fusion_predict(audio_feat, raw_transcript)

        results, question_score = process_mmse_question(
            question_type=question_type,
            transcript=normalized_transcript,  # use normalized for correctness
            ml_prediction=ml_label,
            caregiver_is_correct=caregiver_is_correct
        )

        question_data = {
            "question_type": question_type,
            "transcript_raw": raw_transcript,
            "transcript_normalized": normalized_transcript,
            "ml_prediction": ml_label,
            "ml_probability": ml_prob,
            "caregiver_is_correct": caregiver_is_correct,
            "results": results,
            "question_score": question_score,
            "timestamp": datetime.utcnow()
        }

        await assessments_collection.update_one(
            {"_id": ObjectId(assessment_id), "user_id": user_id},
            {"$push": {"questions": question_data}}
        )

        return {
            "question_score": question_score,
            "ml_prediction": ml_label,
            "ml_probability": ml_prob
        }

    finally:
        os.remove(input_path)


# FINALIZE MMSE
@app.post("/mmse/finalize")
async def finalize_mmse(assessment_id: str, user_id: str):

    assessment = await assessments_collection.find_one({
        "_id": ObjectId(assessment_id),
        "user_id": user_id
    })

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    questions = assessment.get("questions", [])

    total_score = sum(q["question_score"] for q in questions)
    avg_ml_prob = sum(q["ml_probability"] for q in questions) / len(questions)

    ml_risk_label = "Dementia" if avg_ml_prob >= 0.60 else "Control"

    await assessments_collection.update_one(
        {"_id": ObjectId(assessment_id)},
        {
            "$set": {
                "total_score": total_score,
                "ml_summary": {
                    "avg_probability": avg_ml_prob,
                    "ml_risk_label": ml_risk_label
                },
                "status": "completed",
                "completed_at": datetime.utcnow()
            }
        }
    )

    return {
        "total_score": total_score,
        "ml_risk_label": ml_risk_label,
        "avg_ml_probability": avg_ml_prob
    }