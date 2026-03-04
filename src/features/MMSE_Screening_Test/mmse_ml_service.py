# src/features/mmse_screening/mmse_ml_service.py

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import whisper
from dotenv import load_dotenv

load_dotenv()


class MMSEModels:
    def __init__(self):
       # --------------------------------------------------
        # Resolve project root dynamically
        # --------------------------------------------------
        BASE_DIR = Path(__file__).resolve()

        while BASE_DIR.name != "dementia_backend":
            BASE_DIR = BASE_DIR.parent

        # --------------------------------------------------
        # Resolve model directory
        # --------------------------------------------------
        env_model_path = os.getenv("MODEL_PATH")

        if env_model_path:
            model_base = Path(env_model_path)
        else:
            model_base = BASE_DIR / "src"/"models" / "MMSE_Screening_Test"

        if not model_base.exists():
            raise FileNotFoundError(
                f"MMSE model directory not found: {model_base}"
            )

        print(f"[MMSE] Loading models from: {model_base}")

        self.audio_model = joblib.load(os.path.join(model_base, "best_audio_model.pkl"))
        self.audio_scaler = joblib.load(os.path.join(model_base, "audio_scaler.pkl"))

        self.text_model = joblib.load(os.path.join(model_base, "best_text_model.pkl"))
        self.text_tfidf = joblib.load(os.path.join(model_base, "text_tfidf.pkl"))
        self.text_pca = joblib.load(os.path.join(model_base, "text_pca.pkl"))

        whisper_model_name = os.getenv("WHISPER_MODEL", "small")
        self.whisper_model = whisper.load_model(whisper_model_name)


@lru_cache()
def get_mmse_models() -> MMSEModels:
    """Lazy singleton – loads models only once per process."""
    return MMSEModels()


def transcribe_audio(path: str) -> str:
    models = get_mmse_models()
    result = models.whisper_model.transcribe(
        path,
        language="en",
        fp16=False
    )
    return result.get("text", "").strip().lower()


def late_fusion_predict(audio_features: np.ndarray, transcript: str):
    """
    Combine audio and text probabilities into a single dementia risk score.
    """
    models = get_mmse_models()

    # 1. Audio probability
    scaled_audio = models.audio_scaler.transform(audio_features)
    audio_prob = models.audio_model.predict_proba(scaled_audio)[0][1]

    # 2. Text probability (if transcript long enough)
    if len(transcript.split()) < 5:
        fused_prob = audio_prob
    else:
        tfidf_vec = models.text_tfidf.transform([transcript]).toarray()
        text_vec = models.text_pca.transform(tfidf_vec)
        text_prob = models.text_model.predict_proba(text_vec)[0][1]
        fused_prob = 0.6 * audio_prob + 0.4 * text_prob

    label = "Dementia" if fused_prob >= 0.60 else "Control"
    return label, float(fused_prob), float(audio_prob)