from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import librosa
import joblib
import whisper
import tempfile
import os
import re
import subprocess
import uuid
from services.mmse_question_service import process_mmse_question

app = FastAPI(title="AI Dementia Detection API")


audio_model = joblib.load("models/best_audio_model.pkl")
audio_scaler = joblib.load("models/audio_scaler.pkl")

text_model = joblib.load("models/best_text_model.pkl")
text_tfidf = joblib.load("models/text_tfidf.pkl")
text_pca = joblib.load("models/text_pca.pkl")

whisper_model = whisper.load_model("small")


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

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

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
        fp16=False,
        initial_prompt="The patient will say words like ball, car, man, wristwatch, pen."
    )

    text = result.get("text", "")
    return text.strip().lower() if text else ""



def late_fusion_predict(audio_feat, transcript):
    audio_feat = audio_scaler.transform(audio_feat)
    audio_prob = audio_model.predict_proba(audio_feat)[0][1]

      # Short utterance → disable text model
    if len(transcript.split()) < 5:
        fused = audio_prob
    else:
        tfidf_vec = text_tfidf.transform([transcript]).toarray()
        text_vec = text_pca.transform(tfidf_vec)
        text_prob = text_model.predict_proba(text_vec)[0][1]
        fused = 0.6 * audio_prob + 0.4 * text_prob

    return fused >= 0.60 and "Dementia" or "Control", fused


# MMSE endpoint

@app.get("/")
async def root():
    return {"message": "AI Dementia Detection API is running."}

@app.post("/mmse/question")
async def mmse_question(
    question_type: str = Form(...),
    file: UploadFile = File(...),
    caregiver_is_correct: bool = Form(None)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        wav_path = convert_to_wav(path)
        audio_feat = extract_audio_features(wav_path)
        transcript = transcribe_audio(wav_path)
        ml_label, ml_prob = late_fusion_predict(audio_feat, transcript)

        results, total_score = process_mmse_question(
        question_type=question_type,
            transcript=transcript,
            ml_prediction=ml_label,
            caregiver_is_correct=caregiver_is_correct
        )
        return {
            "question_type": question_type,
            "transcript": transcript,
            "ml_prediction": ml_label,
            "ml_probability": ml_prob,
            "results": results,
            "total_score": total_score
        }

    finally:
        os.remove(path)
