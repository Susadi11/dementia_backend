


# src/features/mmse_screening/mmse_ml_service.py

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import whisper
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

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

        def _load_hf_or_local(repo_id: str, filename: str, local_path: str):
            """
            Try to download model from Hugging Face,
            if fails → load from local storage
            """
            try:
                print(f"[MMSE] Downloading {filename} from Hugging Face: {repo_id}")
                hf_path = hf_hub_download(repo_id=repo_id, filename=filename)
                return joblib.load(hf_path)
            except Exception as e:
                print(f"[MMSE] Warning: Could not download {filename} from HF ({e}). Using local fallback.")
                if os.path.exists(local_path):
                    return joblib.load(local_path)
                else:
                    raise FileNotFoundError(f"Neither HF nor local model found: {local_path}")

        # 1. Load Audio Model + Scaler
        self.audio_model = _load_hf_or_local(
            "katharushimethmini02/best_audio_model", 
            "best_audio_model.pkl", 
            os.path.join(model_base, "best_audio_model.pkl")
        )

        # Scaler for normalizing audio features
        self.audio_scaler = _load_hf_or_local(
            "katharushimethmini02/audio_scaler", 
            "audio_scaler.pkl", 
            os.path.join(model_base, "audio_scaler.pkl")
        )

        # 2.  # Scaler for normalizing audio features
        self.text_model = _load_hf_or_local(
            "katharushimethmini02/best_text_model", 
            "best_text_model.pkl", 
            os.path.join(model_base, "best_text_model.pkl")
        )

        #Converts text → numerical features
        self.text_tfidf = _load_hf_or_local(
            "katharushimethmini02/text_tfidf", 
            "text_tfidf.pkl", 
            os.path.join(model_base, "text_tfidf.pkl")
        )

        # Reduces dimensionality of TF-IDF features
        self.text_pca = _load_hf_or_local(
            "katharushimethmini02/text_pca", 
            "text_pca.pkl", 
            os.path.join(model_base, "text_pca.pkl")
        )
        #Load Whisper ASR Model- Used for speech-to-text conversion
        whisper_model_name = os.getenv("WHISPER_MODEL", "small")
        self.whisper_model = whisper.load_model(whisper_model_name)

# ----------------------------------------
# Speech-to-Text using Whisper
# ----------------------------------------
@lru_cache()
def get_mmse_models() -> MMSEModels:
    """Lazy singleton – loads models only once per process."""
    return MMSEModels()

# ----------------------------------------
# Speech-to-Text using Whisper
# ----------------------------------------

def transcribe_audio(path: str) -> str:
    models = get_mmse_models()
     # Convert audio → text
    result = models.whisper_model.transcribe(
        path,
        language="en",
        fp16=False
    )
    return result.get("text", "").strip().lower()

# ----------------------------------------
# Late Fusion Prediction
# ----------------------------------------
def late_fusion_predict(audio_features: np.ndarray, transcript: str):
    """
    Combine audio and text probabilities into a single dementia risk score.
    """
    models = get_mmse_models()

    # ----------------------------------------
    # Step 1: Audio Prediction
    # ----------------------------------------

    scaled_audio = models.audio_scaler.transform(audio_features)
    audio_prob = models.audio_model.predict_proba(scaled_audio)[0][1]

    # ----------------------------------------
    # Step 1: Audio Prediction
    # ----------------------------------------

    # If transcript is too short → skip text model
    if len(transcript.split()) < 5:
        fused_prob = audio_prob
    else:
        # Convert text → TF-IDF features
        tfidf_vec = models.text_tfidf.transform([transcript]).toarray()
        # Reduce dimensions using PCA
        text_vec = models.text_pca.transform(tfidf_vec)
        # Predict text probability
        text_prob = models.text_model.predict_proba(text_vec)[0][1]
        # ----------------------------------------
        # Step 3: Fusion (Weighted Average)
        # ----------------------------------------
        fused_prob = 0.6 * audio_prob + 0.4 * text_prob
    # ----------------------------------------
    # Step 3: Fusion (Weighted Average)
    # ----------------------------------------
    label = "Dementia" if fused_prob >= 0.60 else "Control"
    return label, float(fused_prob), float(audio_prob)
